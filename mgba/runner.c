/* Minimal uncapped mGBA 0.10.5 host for the same Lua bridge used by the GUI. */
#include <mgba/flags.h>
#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/log.h>
#include <mgba/core/scripting.h>
#include <mgba/script/context.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

static volatile sig_atomic_t stopping;
static void stop(int signal_number) { (void)signal_number; stopping = 1; }
static void log_message(struct mLogger* logger, int category, enum mLogLevel level,
                        const char* format, va_list args) {
    (void)logger;
    if (!(level & (mLOG_FATAL | mLOG_ERROR | mLOG_WARN)) &&
        !(category == mLogCategoryById("script") && level == mLOG_INFO)) return;
    fprintf(stderr, "%s: ", mLogCategoryName(category));
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s ROM LUA_SCRIPT [FRAMES; 0=until SIGINT]\n", argv[0]);
        return 2;
    }
    unsigned long limit = 0;
    if (argc == 4) {
        char* end;
        errno = 0;
        limit = strtoul(argv[3], &end, 10);
        if (errno || !*argv[3] || *end || *argv[3] == '-') return 2;
    }
    struct mLogger logger = { .log = log_message, .filter = NULL };
    mLogSetDefaultLogger(&logger);
    struct mCore* core = mCoreFind(argv[1]);
    if (!core) { fprintf(stderr, "Unsupported or missing ROM\n"); return 1; }
    if (!core->init(core)) { fprintf(stderr, "Core initialization failed\n"); return 1; }
    mCoreConfigInit(&core->config, "rogue-rl");
    struct mCoreOptions options = {0};
    options.audioSync = false;
    options.videoSync = false;
    mCoreConfigLoadDefaults(&core->config, &options);
    mCoreConfigSetDefaultValue(&core->config, "idleOptimization", "detect");
    mCoreLoadConfig(core);
    int status = 1;
    color_t* pixels = calloc(256 * 224, sizeof(*pixels));
    if (!pixels) goto cleanup;
    core->setVideoBuffer(core, pixels, 256);
    if (!mCoreLoadFile(core, argv[1])) { fprintf(stderr, "ROM load failed\n"); goto cleanup; }
    core->reset(core);
    struct mScriptContext context;
    mScriptContextInit(&context);
    mScriptContextAttachStdlib(&context);
    mScriptContextAttachSocket(&context);
    mScriptContextRegisterEngines(&context);
    mScriptContextAttachLogger(&context, &logger);
    mScriptContextAttachCore(&context, core);
    struct mScriptEngineContext* engine = HashTableLookup(&context.engines, "lua");
    if (engine && mScriptContextLoadFile(&context, argv[2]) && engine->run(engine)) {
        signal(SIGINT, stop);
        signal(SIGTERM, stop);
        unsigned long frames = 0;
        while (!stopping && (!limit || frames < limit)) {
            core->runFrame(core);
            mScriptContextTriggerCallback(&context, "frame");
            ++frames;
        }
        fprintf(stderr, "Completed %lu frames\n", frames);
        status = 0;
    } else {
        fprintf(stderr, "Lua script failed to load\n");
        if (engine && engine->getError(engine)) fprintf(stderr, "%s\n", engine->getError(engine));
    }
    mScriptContextDetachCore(&context);
    mScriptContextDeinit(&context);
cleanup:
    mCoreConfigDeinit(&core->config);
    core->deinit(core);
    free(pixels);
    return status;
}
