"""Execute the real Lua bridge with an in-memory mGBA API fixture."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_lua_protocol_mailbox_and_stages(tmp_path):
    lua = shutil.which("lua")
    if lua is None:
        pytest.skip("Lua interpreter required to execute bridge fixture")
    bridge = Path(__file__).resolve().parents[1] / "mgba/bridge.lua"
    harness = r"""
ROGUE_RL_PROFILE={mailbox_address=0,abi=1,rom_size=512,rom_crc32="12345678",rom_sha256="hash",source_revision="rev",port=8765}
C={CHECKSUM={CRC32=0}}
local mem={[0]=0x524c5247,[1]=1,[2]=0,[3]=0,[4]=0}
emu={read32=function(_,a) return mem[a/4] or 0 end,
write32=function(_,a,v) mem[a/4]=v end,romSize=function() return 512 end,
checksum=function() return string.char(0x12,0x34,0x56,0x78) end,
setKeys=function() end,
loadStateFile=function() mem[0]=0x524c5247;mem[1]=1;mem[2]=0;return true end}
console={log=function() end}
local frame
callbacks={add=function(_,name,fn) frame=fn end}
local receive_callback, accept_callback, queue, outputs = nil,nil,{},{}
local client={}
function client:add(name,fn) if name=="received" then receive_callback=fn end end
function client:receive() if #queue>0 then return table.remove(queue,1) end return nil,"again" end
function client:send(data) table.insert(outputs,data); return #data end
function client:close() end
function client:poll() if #queue>0 then receive_callback() end end
local server={}
function server:listen() return 0 end
function server:add(name,fn) accept_callback=fn end
function server:accept() return client end
function server:poll() if accept_callback then local fn=accept_callback;accept_callback=nil;fn() end end
socket={ERRORS={AGAIN="again"},bind=function() return server end}
dofile(BRIDGE_PATH)
local function core()
 if mem[2]==1 and mem[3]==0 then
  mem[4]=mem[4]+1;mem[3]=1;mem[11]=1
  for _,offset in ipairs({12,28,44,60,76,92,108,124}) do
   mem[offset]=25;mem[offset+1]=24;mem[offset+2]=48
   for i=8,14 do mem[offset+i]=6 end
  end
 end
 if mem[7]==1 then mem[7]=0;mem[4]=mem[4]+1;mem[3]=3;mem[10]=1;mem[11]=0 end
end
local function send(line)
 table.insert(queue,line); frame();core();frame()
end
send("HELLO\t1\n")
send("RESET\t2\t626174746c65\t2f746d702f7374617465\t100\n")
send("STEP\t3\t0\t0\t100\n") -- stale decision must reject
send("STEP\t4\t1\t0\t100\n")
for _,value in ipairs(outputs) do io.write(value) end
""".replace("BRIDGE_PATH", json.dumps(str(bridge)))
    path = tmp_path / "fixture.lua"
    path.write_text(harness)
    result = subprocess.run([lua, str(path)], check=True, capture_output=True, text=True, timeout=5)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 4
    assert rows[0]["result"]["protocol"] == 1
    assert rows[1]["result"]["player"]["stat_stages"] == [6] * 7
    assert rows[1]["result"]["opponent"]["hp_fraction"] == 0.5
    assert "hp" not in rows[1]["result"]["opponent"]
    assert rows[2]["error"] == "stale_decision"
    assert rows[3]["result"]["outcome"] == "win"
    assert rows[3]["result"]["legal_actions"] == [False] * 10
