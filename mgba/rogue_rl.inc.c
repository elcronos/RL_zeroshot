/* Research instrumentation for Pokabbie/pokeemerald-rogue expansion
 * a6adfcf18d7eaf99c2803e4b0bc04eca7af2f014. Included in the player controller
 * after its static declarations, so the existing completion routine is used.
 * ABI consists solely of aligned little-endian u32 words; no guessed offsets.
 */
#include "battle_util.h"
#include "event_data.h"
#include "trainer_see.h"
#include "rogue_settings.h"
#include "rogue_trainers.h"

#define RL_MAGIC 0x524C5247
#define RL_ABI 1
#define RL_RUNNING 0
#define RL_ACTION 1
#define RL_SWITCH 2
#define RL_TERMINAL 3
#define RL_ERROR 4

struct RogueRLMailbox
{
    u32 magic, abi, enabled, phase, decision;
    u32 actionDecision, action, actionReady, error, turn, outcome, legal;
    u32 player[16], opponent[16], party[6][16], moves[4][8];
};
EWRAM_DATA volatile struct RogueRLMailbox gRogueRL = {0};
static EWRAM_DATA bool8 sRogueRLSwitchPending = FALSE;

/* Called only on the first warp into Rogue_HubTransition, after BeginRogueRun.
 * The map's own entry script consumes the one-shot marker and uses Rogue's
 * normal trainerbattle command. This is a real route trainer with the run's
 * current encounter filters and party generation, not a synthetic fixture.
 * 0x40D1 is declared unused in this pinned source revision. */
void RogueRLPrepareBattle(void)
{
    u16 trainer = TRAINER_NONE;
    u32 i;
    bool32 hasUsableMon = FALSE;

    VarSet(VAR_UNUSED_0x40D1, 0);
    if (!Rogue_IsRunActive())
        return;
    for (i = 0; i < gPlayerPartyCount; ++i)
    {
        if (GetMonData(&gPlayerParty[i], MON_DATA_HP) != 0)
        {
            hasUsableMon = TRUE;
            break;
        }
    }
    if (!hasUsableMon)
        return;

    Rogue_ChooseRouteTrainers(&trainer, 1);
    if (trainer == TRAINER_NONE || trainer >= TRAINERS_COUNT)
        return;

    /* The battle bridge supports trainer singles with SET prompts disabled. */
    Rogue_SetConfigToggle(CONFIG_TOGGLE_SWITCH_MODE, FALSE);
    gNoOfApproachingTrainers = 0;
    VarSet(VAR_ROGUE_SPECIAL_ENCOUNTER_DATA, trainer);
    VarSet(VAR_UNUSED_0x40D1, 1);
}

/* Six reproducible three-Pokémon teams for bootstrapping a diverse battle
 * corpus from a new game's lab. The current game RNG selects one team after
 * the title-screen delay; the chosen team and state hash are captured as
 * provenance. The research entry script immediately warps to the normal run
 * transition; route trainer selection and battle setup remain untouched.
 * 0x40D2 is also unused in the pinned source revision. */
void RogueRLPrepareFixture(void)
{
    static const u16 sTeams[][3] =
    {
        {SPECIES_BULBASAUR, SPECIES_PIDGEY, SPECIES_PIKACHU},
        {SPECIES_CHARMANDER, SPECIES_ODDISH, SPECIES_GEODUDE},
        {SPECIES_SQUIRTLE, SPECIES_ABRA, SPECIES_MACHOP},
        {SPECIES_CHIKORITA, SPECIES_HOOTHOOT, SPECIES_MAREEP},
        {SPECIES_TORCHIC, SPECIES_LOTAD, SPECIES_ARON},
        {SPECIES_MUDKIP, SPECIES_RALTS, SPECIES_NUMEL},
    };
    u32 i, team;
    if (Rogue_IsRunActive() || gPlayerPartyCount != 0 || FlagGet(FLAG_SYS_POKEMON_GET))
        return;
    team = Random() % ARRAY_COUNT(sTeams);
    for (i = 0; i < ARRAY_COUNT(sTeams[team]); ++i)
        CreateMon(&gPlayerParty[i], sTeams[team][i], STARTER_MON_LEVEL,
            USE_RANDOM_IVS, FALSE, 0, OT_ID_PLAYER_ID, 0);
    CalculatePlayerPartyCount();
    FlagSet(FLAG_SYS_POKEMON_GET);
    VarSet(VAR_UNUSED_0x40D2, 1);
}

/* Suppress frame-clock RNG cycling while research is enabled. Actual battle
 * Random() draws are untouched, so inference latency cannot change outcomes. */
bool32 RogueRLIsEnabled(void)
{
    return gRogueRL.magic == RL_MAGIC && gRogueRL.enabled;
}

static void RogueRLFail(u32 error)
{
    gRogueRL.error = error;
    gRogueRL.phase = RL_ERROR;
}

static bool32 RogueRLSupported(u32 battler)
{
    /* Exactly ordinary trainer singles during a Rogue run. */
    u32 allowed = BATTLE_TYPE_TRAINER | BATTLE_TYPE_IS_MASTER | BATTLE_TYPE_LINK_IN_BATTLE;
    if (!Rogue_IsRunActive() || !(gBattleTypeFlags & BATTLE_TYPE_TRAINER)
        || (gBattleTypeFlags & ~allowed) || gBattlersCount != 2
        || GetBattlerSide(battler) != B_SIDE_PLAYER
        || gBattleScripting.battleStyle != OPTIONS_BATTLE_STYLE_SET
        || IsCurseActive(EFFECT_AUTO_MOVE_SELECT)
        || Rogue_GetActiveCampaign() == ROGUE_CAMPAIGN_AUTO_BATTLER)
    {
        RogueRLFail(1);
        return FALSE;
    }
    return TRUE;
}

static void RogueRLReadMon(u32 battler, volatile u32 *dst, bool32 opponent)
{
    u32 i, species = gBattleMons[battler].species;
    struct Pokemon *illusion = opponent ? GetIllusionMonPtr(battler) : NULL;
    if (illusion != NULL)
        species = GetMonData(illusion, MON_DATA_SPECIES);
    dst[0] = species;
    /* Enemy HP is the 48-pixel public health bar, never exact enemy HP/stats. */
    dst[1] = opponent ? (gBattleMons[battler].hp == 0 ? 0 :
        (gBattleMons[battler].hp * 48 / gBattleMons[battler].maxHP == 0 ? 1 :
         gBattleMons[battler].hp * 48 / gBattleMons[battler].maxHP)) : gBattleMons[battler].hp;
    dst[2] = opponent ? 48 : gBattleMons[battler].maxHP;
    dst[3] = illusion ? GetMonData(illusion, MON_DATA_LEVEL) : gBattleMons[battler].level;
    dst[4] = gBattleMons[battler].status1;
    /* Opponent types are public species typing, not private transformed typing. */
    dst[5] = opponent ? GetTypeBySpecies(species, 0, 0) : gBattleMons[battler].type1;
    dst[6] = opponent ? GetTypeBySpecies(species, 1, 0) : gBattleMons[battler].type2;
    dst[7] = opponent ? 0 : gBattleMons[battler].speed;
    for (i = 0; i < 7; ++i)
        dst[8 + i] = gBattleMons[battler].statStages[STAT_ATK + i];
    dst[15] = 1;
}

static void RogueRLPublish(u32 battler, u32 phase)
{
    u32 i, j, move, unusable = 15, canSwitch;
    gRogueRL.legal = 0;
    gRogueRL.turn = gBattleResults.battleTurnCounter;
    gRogueRL.outcome = 0;
    RogueRLReadMon(battler, gRogueRL.player, FALSE);
    RogueRLReadMon(BATTLE_OPPOSITE(battler), gRogueRL.opponent, TRUE);
    if (phase == RL_ACTION)
    {
        /* Same engine mask includes PP, Disable, Encore, Choice, Taunt, etc. */
        unusable = CheckMoveLimitations(battler, 0, MOVE_LIMITATIONS_ALL);
        gRogueRL.legal = (~unusable) & 15;
        if (unusable == 15)
            gRogueRL.legal = 1; /* slot zero requests engine's Struggle branch */
    }
    canSwitch = phase == RL_SWITCH || (CanBattlerEscape(battler)
        && (ItemId_GetHoldEffect(gBattleMons[battler].item) == HOLD_EFFECT_SHED_SHELL
            || !IsAbilityPreventingEscape(battler)));
    for (i = 0; i < PARTY_SIZE; ++i)
    {
        volatile u32 *dst = gRogueRL.party[i];
        struct Pokemon *mon = &gPlayerParty[i];
        dst[0] = GetMonData(mon, MON_DATA_SPECIES);
        dst[1] = GetMonData(mon, MON_DATA_HP);
        dst[2] = GetMonData(mon, MON_DATA_MAX_HP);
        dst[3] = GetMonData(mon, MON_DATA_LEVEL);
        dst[4] = GetMonData(mon, MON_DATA_STATUS);
        dst[5] = GetTypeBySpecies(dst[0], 0, GetMonData(mon, MON_DATA_OT_ID));
        dst[6] = GetTypeBySpecies(dst[0], 1, GetMonData(mon, MON_DATA_OT_ID));
        dst[7] = GetMonData(mon, MON_DATA_SPEED);
        for (j = 8; j < 15; ++j)
            dst[j] = 6;
        dst[15] = i == gBattlerPartyIndexes[battler];
        if (canSwitch && i < gPlayerPartyCount && dst[0] != SPECIES_NONE
            && dst[1] != 0 && !dst[15] && !GetMonData(mon, MON_DATA_IS_EGG))
            gRogueRL.legal |= 1 << (4 + i);
    }
    for (i = 0; i < MAX_MON_MOVES; ++i)
    {
        volatile u32 *dst = gRogueRL.moves[i];
        move = gBattleMons[battler].moves[i];
        if (phase == RL_ACTION && unusable == 15 && i == 0)
            move = MOVE_STRUGGLE;
        dst[0] = move;
        dst[1] = gBattleMoves[move].power;
        dst[2] = gBattleMoves[move].type;
        dst[3] = gBattleMons[battler].pp[i];
        dst[4] = CalculatePPWithBonus(move, gBattleMons[battler].ppBonuses, i);
        dst[5] = gBattleMoves[move].accuracy;
        dst[6] = gBattleMoves[move].split;
        dst[7] = move == MOVE_STRUGGLE && unusable == 15;
    }
    gRogueRL.actionReady = 0;
    ++gRogueRL.decision;
    gRogueRL.phase = phase; /* publish last */
}

static bool32 RogueRLConsume(void)
{
    if (!gRogueRL.actionReady)
        return FALSE;
    gRogueRL.actionReady = 0;
    if (gRogueRL.actionDecision != gRogueRL.decision)
    {
        RogueRLFail(2);
        return FALSE;
    }
    if (gRogueRL.action >= 10 || !(gRogueRL.legal & (1 << gRogueRL.action)))
    {
        RogueRLFail(3);
        return FALSE;
    }
    gRogueRL.phase = RL_RUNNING;
    return TRUE;
}

static bool32 RogueRLChooseAction(u32 battler)
{
    if (!gRogueRL.enabled)
    {
        /* Publish a non-invasive capture marker at the first action menu.
         * The original player controller still handles the command. A Lua
         * harness can save a clean pre-policy state while enabled is zero. */
        if (gRogueRL.magic == RL_MAGIC && gRogueRL.phase == RL_RUNNING
            && Rogue_IsRunActive() && (gBattleTypeFlags & BATTLE_TYPE_TRAINER)
            && !(gBattleTypeFlags & ~(BATTLE_TYPE_TRAINER | BATTLE_TYPE_IS_MASTER | BATTLE_TYPE_LINK_IN_BATTLE))
            && gBattlersCount == 2 && GetBattlerSide(battler) == B_SIDE_PLAYER
            && gBattleScripting.battleStyle == OPTIONS_BATTLE_STYLE_SET)
            gRogueRL.phase = RL_ACTION;
        return FALSE;
    }
    if (!RogueRLSupported(battler) || gRogueRL.phase == RL_ERROR)
        return TRUE;
    if (gRogueRL.phase == RL_RUNNING)
        RogueRLPublish(battler, RL_ACTION);
    if (RogueRLConsume())
    {
        sRogueRLSwitchPending = gRogueRL.action >= 4;
        BtlController_EmitTwoReturnValues(battler, BUFFER_B,
            sRogueRLSwitchPending ? B_ACTION_SWITCH : B_ACTION_USE_MOVE, 0);
        PlayerBufferExecCompleted(battler);
    }
    return TRUE;
}

static bool32 RogueRLChooseMove(u32 battler)
{
    u32 move, target;
    if (!gRogueRL.enabled)
        return FALSE;
    if (gRogueRL.action >= 4 || gRogueRL.phase == RL_ERROR)
    {
        RogueRLFail(4);
        return TRUE;
    }
    move = gBattleMons[battler].moves[gRogueRL.action];
    target = GetBattlerMoveTargetType(battler, move) & MOVE_TARGET_USER ? battler : BATTLE_OPPOSITE(battler);
    BtlController_EmitTwoReturnValues(battler, BUFFER_B, 10, gRogueRL.action | (target << 8));
    PlayerBufferExecCompleted(battler);
    return TRUE;
}

static bool32 RogueRLChoosePokemon(u32 battler)
{
    u32 i, mode;
    if (!gRogueRL.enabled)
        return FALSE;
    if (!RogueRLSupported(battler) || gRogueRL.phase == RL_ERROR)
        return TRUE;
    mode = gBattleResources->bufferA[battler][1] & 15;
    if (mode != PARTY_ACTION_CHOOSE_MON && mode != PARTY_ACTION_SEND_OUT)
    {
        RogueRLFail(5); /* Never silently bypass an engine switch restriction. */
        return TRUE;
    }
    if (!sRogueRLSwitchPending)
    {
        if (gRogueRL.phase == RL_RUNNING)
            RogueRLPublish(battler, RL_SWITCH);
        if (!RogueRLConsume())
            return TRUE;
    }
    sRogueRLSwitchPending = FALSE;
    for (i = 0; i < ARRAY_COUNT(gBattlePartyCurrentOrder); ++i)
        gBattlePartyCurrentOrder[i] = gBattleResources->bufferA[battler][4 + i];
    BtlController_EmitChosenMonReturnValue(battler, BUFFER_B, gRogueRL.action - 4, gBattlePartyCurrentOrder);
    PlayerBufferExecCompleted(battler);
    return TRUE;
}

/* Called before BattleMainCB2 accesses or frees battle resources. */
bool32 RogueRLTick(void)
{
    if (gRogueRL.magic != RL_MAGIC)
    {
        gRogueRL.magic = RL_MAGIC;
        gRogueRL.abi = RL_ABI;
        gRogueRL.enabled = 0;
        gRogueRL.phase = RL_RUNNING;
        sRogueRLSwitchPending = FALSE;
    }
    if (!gRogueRL.enabled)
        return FALSE;
    if (gRogueRL.phase == RL_ERROR || gRogueRL.phase == RL_TERMINAL)
        return TRUE;
    if ((gRogueRL.phase == RL_ACTION || gRogueRL.phase == RL_SWITCH) && !gRogueRL.actionReady)
        return TRUE;
    if (gBattleOutcome)
    {
        if (gBattleOutcome != B_OUTCOME_WON && gBattleOutcome != B_OUTCOME_LOST && gBattleOutcome != B_OUTCOME_DREW)
        {
            RogueRLFail(6);
            return TRUE;
        }
        gRogueRL.outcome = gBattleOutcome;
        gRogueRL.legal = 0;
        ++gRogueRL.decision;
        gRogueRL.phase = RL_TERMINAL;
        return TRUE;
    }
    return FALSE;
}
