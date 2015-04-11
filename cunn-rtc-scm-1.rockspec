package = "cunn-rtc"
version = "scm-1"

source = {
   url = "git://github.com/szagoruyko/cunn-rtc.git"
}

description = {
   summary = "cunn realtime compiled modules",
   detailed = [[
   ]],
   homepage = "https://github.com/szagoruyko/cunn-rtc",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "cutorch-rtc",
   "nn"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
