task("prepare-lc")
    set_category("plugin")
    on_run("main")
    set_menu {
        usage = "xmake prepare-lc",
        description = "Clone or safely fast-forward LuisaCompute and initialize its submodules",
        options = {},
    }
task_end()
