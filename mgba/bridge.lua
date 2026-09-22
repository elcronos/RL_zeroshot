-- Load the generated profile.lua launcher from mGBA Tools > Scripting.
-- mGBA >=0.10; API source: https://mgba.io/docs/scripting.html
local profile = assert(ROGUE_RL_PROFILE, "Load generated profile.lua, not bridge.lua directly")
local base = assert(profile.mailbox_address)
local MAGIC = 0x524c5247
local MAX_LINE = 16384
local NULL = {}
local client, pending, rx, tx, last_seq, battle_id, ready_decision, disconnect
rx, tx, last_seq = "", "", 0

local function json(value)
    if value == NULL then return "null" end
    local kind = type(value)
    if kind == "boolean" or kind == "number" then return tostring(value) end
    if kind == "string" then
        return '"' .. value:gsub('[%z\1-\31\\"]', function(c)
            return string.format("\\u%04x", string.byte(c))
        end) .. '"'
    end
    local chunks = {}
    if #value > 0 then
        for i, item in ipairs(value) do chunks[i] = json(item) end
        return "[" .. table.concat(chunks, ",") .. "]"
    end
    for key, item in pairs(value) do table.insert(chunks, json(key) .. ":" .. json(item)) end
    return "{" .. table.concat(chunks, ",") .. "}"
end
local function read(word) return emu:read32(base + 4 * word) end
local function write(word, value) emu:write32(base + 4 * word, value) end
local function reply(seq, result, err)
    tx = tx .. json({seq=seq, result=result or NULL, error=err or NULL}) .. "\n"
end
local function crc_hex()
    return (emu:checksum(C.CHECKSUM.CRC32):gsub(".", function(c)
        return string.format("%02x", string.byte(c))
    end))
end
local function check_rom()
    return emu:romSize() == profile.rom_size and crc_hex() == profile.rom_crc32
end
local function unhex(value)
    if not value or #value % 2 ~= 0 or value:find("[^0-9a-f]") then return nil end
    return (value:gsub("..", function(pair) return string.char(tonumber(pair, 16)) end))
end
local function integer(value, lo, hi)
    local n = tonumber(value)
    if not n or n ~= math.floor(n) or n < lo or n > hi then return nil end
    return n
end
local function mon(offset, opponent)
    local result = {
        species=read(offset), level=read(offset+3), status=read(offset+4),
        type_ids={read(offset+5), read(offset+6)}, stat_stages={}, active=read(offset+15) == 1,
    }
    for i=1,7 do result.stat_stages[i] = read(offset+7+i) end
    if opponent then result.hp_fraction = read(offset+1) / math.max(1,read(offset+2))
    else
        result.hp, result.max_hp, result.speed = read(offset+1), read(offset+2), read(offset+7)
    end
    return result
end
local function observation()
    local phase = read(3)
    local result = {
        battle_id=battle_id, decision_id=read(4), turn=read(9),
        phase=({[1]="action", [2]="forced_switch", [3]="terminal"})[phase],
        player=mon(12,false), opponent=mon(28,true), party={}, moves={}, legal_actions={},
        outcome=({[1]="win",[2]="loss",[3]="draw"})[read(10)] or NULL,
    }
    local legal = read(11)
    for i=1,10 do result.legal_actions[i] = math.floor(legal / 2^(i-1)) % 2 == 1 end
    for i=1,6 do result.party[i] = mon(44+(i-1)*16,false) end
    for i=1,4 do
        local o = 140+(i-1)*8
        result.moves[i] = {id=read(o),power=read(o+1),type=read(o+2),pp=read(o+3),
            max_pp=read(o+4),accuracy=read(o+5),category=read(o+6),struggle=read(o+7)==1}
    end
    return result
end
local function command(line)
    local fields = {}
    for field in (line.."\t"):gmatch("(.-)\t") do table.insert(fields,field) end
    local op, seq = fields[1], integer(fields[2],1,2147483647)
    if not seq then reply(0,nil,"invalid_request_sequence"); return end
    if seq <= last_seq then reply(seq,nil,"stale_request_sequence"); return end
    last_seq = seq
    -- mGBA 0.10's socket wrapper can report a closed peer as AGAIN. An
    -- explicit close allows another run to connect to the same emulator.
    if op == "CLOSE" then disconnect(); return end
    if pending then reply(seq,nil,"busy"); return end
    if op == "HELLO" then
        if not check_rom() then reply(seq,nil,"rom_crc_mismatch"); return end
        reply(seq,{protocol=1,abi=profile.abi,rom_sha256=profile.rom_sha256,
            rom_crc32=profile.rom_crc32,source_revision=profile.source_revision})
    elseif op == "RESET" then
        local id,path,limit = unhex(fields[3]),unhex(fields[4]),integer(fields[5],1,10000000)
        if not id or id=="" or not path or path:find("%z") or not limit then
            reply(seq,nil,"invalid_reset"); return
        end
        if not check_rom() then reply(seq,nil,"rom_crc_mismatch"); return end
        emu:setKeys(0)
        if not emu:loadStateFile(path,31) then reply(seq,nil,"savestate_load_failed"); return end
        if read(0) ~= MAGIC or read(1) ~= profile.abi then
            reply(seq,nil,"research_mailbox_missing_or_wrong_abi"); return
        end
        if read(2) ~= 0 then
            reply(seq,nil,"savestate_must_be_captured_before_research_enabled"); return
        end
        battle_id,ready_decision = id,nil
        write(2,1); write(3,0); write(4,0); write(7,0); write(8,0); write(10,0)
        pending={seq=seq,frames=0,limit=limit,after=-1}
    elseif op == "STEP" then
        local decision=integer(fields[3],0,4294967295)
        local action=integer(fields[4],0,9)
        local limit=integer(fields[5],1,10000000)
        if not battle_id or not action or not decision or not limit then reply(seq,nil,"invalid_step"); return end
        if decision ~= ready_decision or decision ~= read(4) then reply(seq,nil,"stale_decision"); return end
        if read(3) ~= 1 and read(3) ~= 2 then reply(seq,nil,"not_at_decision"); return end
        if math.floor(read(11)/2^action)%2 ~= 1 then reply(seq,nil,"illegal_action"); return end
        write(5,decision); write(6,action); write(7,1)
        ready_decision=nil
        pending={seq=seq,frames=0,limit=limit,after=decision}
    elseif op == "SCREENSHOT" then
        local path=unhex(fields[3])
        if not battle_id or not path or path=="" or path:find("%z") then
            reply(seq,nil,"invalid_screenshot"); return
        end
        -- The controller supplies an absolute workspace path and verifies the
        -- created file before reporting a visual frame as evidence.
        emu:screenshot(path)
        reply(seq,{path=path,decision_id=read(4)})
    elseif op == "MENU_SCREENSHOT" then
        local path,scratch=unhex(fields[3]),unhex(fields[4])
        if not battle_id or not path or not scratch or path=="" or scratch==""
            or read(3) ~= 1 or ready_decision ~= read(4) then
            reply(seq,nil,"menu_screenshot_requires_action_decision"); return
        end
        -- Direct mailbox selection intentionally skips this UI. Save the exact
        -- boundary, let the unmodified game open Fight once, capture its own
        -- move menu, then restore the boundary before answering the controller.
        if not emu:saveStateFile(scratch,31) then reply(seq,nil,"menu_savestate_failed"); return end
        write(2,0)
        pending={seq=seq,frames=0,limit=40,kind="menu",path=path,scratch=scratch}
    else reply(seq,nil,"unknown_command") end
end
disconnect = function()
    if client then client:close() end
    client,pending,ready_decision=nil,nil,nil
    rx,tx,last_seq="","",0
    emu:setKeys(0)
end
local function receive()
    while client do
        local data,err=client:receive(4096)
        if not data then
            if err ~= socket.ERRORS.AGAIN then disconnect() end
            return
        end
        if data == "" then disconnect(); return end
        rx=rx..data
        if #rx>MAX_LINE then disconnect(); return end
        while true do
            local ending=rx:find("\n",1,true)
            if not ending then break end
            local line=rx:sub(1,ending-1)
            rx=rx:sub(ending+1)
            command(line)
        end
    end
end
local server,err=socket.bind("127.0.0.1",profile.port)
assert(server,err)
assert(server:listen())
server:add("received",function()
    local candidate=server:accept()
    if not candidate then return end
    if client then candidate:close(); return end
    client=candidate
    client:add("received",receive)
    client:add("error",disconnect)
end)
callbacks:add("frame",function()
    server:poll()
    if client then client:poll() end
    if pending then
        pending.frames=pending.frames+1
        local phase=read(3)
        if pending.kind == "menu" then
            if pending.frames >= pending.limit then
                emu:setKeys(0)
                emu:screenshot(pending.path)
                if not emu:loadStateFile(pending.scratch,31) then
                    reply(pending.seq,nil,"menu_state_restore_failed")
                else
                    reply(pending.seq,{path=pending.path,decision_id=read(4)})
                end
                pending=nil
            end
        elseif phase==4 then
            reply(pending.seq,nil,"rom_hook_error_"..tostring(read(8)))
            pending=nil
        elseif (phase==1 or phase==2 or phase==3) and read(4)>pending.after then
            ready_decision=read(4)
            reply(pending.seq,observation())
            pending=nil
        elseif pending.frames>=pending.limit then
            reply(pending.seq,nil,"frame_limit_exceeded")
            pending=nil
        end
    end
    -- Input is suppressed at policy boundaries; only text/animations are advanced.
    if pending and pending.kind == "menu" and pending.frames >= 16 and pending.frames <= 23 then
        emu:setKeys(1) -- A: choose Fight after the action-menu DMA has settled.
    else
        emu:setKeys(pending and read(3)==0 and pending.frames%4==0 and 1 or 0)
    end
    if client and #tx>0 then
        local sent,senderr=client:send(tx)
        if sent then tx=tx:sub(sent+1)
        elseif senderr ~= socket.ERRORS.AGAIN then disconnect() end
    end
end)
console:log("Rogue RL bridge listening on 127.0.0.1:"..profile.port)
