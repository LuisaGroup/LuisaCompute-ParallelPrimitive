const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');

const root = path.resolve(__dirname, '..');
const options = {
    cccl: path.join(root, 'build/cccl-v3.5.0-full'),
    cuda: '/usr/local/cuda',
    arch: 'sm_120',
    lcpp: path.join(root, 'src'),
    'reuse-cub': '0',
};
for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i].replace(/^--/, '');
    if (!(key in options) || !process.argv[i + 1]) {
        throw new Error('Usage: node benchmarks/rebuild_cuda_cub_misc.cjs [--cccl DIR] [--cuda DIR] [--arch sm_120] [--lcpp DIR] [--reuse-cub 0|1]');
    }
    options[key] = process.argv[i + 1];
}
function run(command, args) {
    const result = spawnSync(command, args, { cwd: root, stdio: 'inherit' });
    if (result.error) throw result.error;
    if (result.status !== 0) process.exit(result.status ?? 1);
}
if (!/^sm_[0-9]+[a-z]?$/.test(options.arch)) throw new Error('Expected an explicit sm_NNN architecture.');
const cccl = path.resolve(root, options.cccl);
const cuda = path.resolve(root, options.cuda);
const lcpp = path.resolve(root, options.lcpp);
if (!fs.existsSync(path.join(lcpp, 'lcpp/parallel_primitive.h'))) throw new Error('Missing LCPP headers: ' + lcpp);
if (!['0', '1'].includes(options['reuse-cub'])) throw new Error('--reuse-cub must be 0 or 1.');
for (const item of ['cub/cub/version.cuh', 'thrust/thrust/version.h', 'libcudacxx/include/cuda/stream_ref']) {
    if (!fs.existsSync(path.join(cccl, item))) throw new Error('Missing CCCL header: ' + item);
}
const compileCommands = path.join(root, 'build/compile_commands.json');
const entries = JSON.parse(fs.readFileSync(compileCommands, 'utf8'));
const entry = entries.find(e => path.basename(e.file) === 'device_scan_test.cpp');
if (!entry || !Array.isArray(entry.arguments)) {
    throw new Error('Generate build/compile_commands.json with xmake project -k compile_commands build first.');
}
const args = [...entry.arguments];
const compiler = args.shift();
const sourceIndex = args.findIndex(arg => arg === entry.file || path.resolve(entry.directory, arg) === path.resolve(entry.directory, entry.file));
const outputIndex = args.indexOf('-o');
if (sourceIndex < 0 || outputIndex < 0) throw new Error('Unrecognized compile command.');
args[sourceIndex] = path.join(root, 'benchmarks/cuda_cub_misc.cpp');
args[outputIndex + 1] = path.join(root, 'build/cuda_cub_misc.o');
args.unshift('-I' + path.join(cuda, 'include'));
args.unshift('-I' + lcpp);
const cubFlags = [
    '-std=c++20', '-O3', '-lineinfo', '-arch=' + options.arch,
    '-I' + path.join(cccl, 'cub'),
    '-I' + path.join(cccl, 'thrust'),
    '-I' + path.join(cccl, 'libcudacxx/include'),
];
const cubArgs = [...cubFlags, '-c', 'benchmarks/cub_misc.cu', '-o', 'build/cub_misc.o'];
const nvcc = path.join(cuda, 'bin/nvcc');
const nvccVersion = spawnSync(nvcc, ['--version'], { encoding: 'utf8' });
if (nvccVersion.status !== 0) throw new Error('Cannot query nvcc version.');
const manifest = spawnSync(nvcc, [...cubFlags, '-M', 'benchmarks/cub_misc.cu'], {
    cwd: root, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024,
});
if (manifest.error) throw manifest.error;
if (manifest.status !== 0) throw new Error('Cannot query CUB dependencies:\n' + manifest.stderr);
const separator = manifest.stdout.indexOf(':');
if (separator < 0) throw new Error('Invalid NVCC dependency manifest.');
// NVCC emits Make prerequisites with escaped whitespace and line continuations.
const prerequisites = manifest.stdout.slice(separator + 1).replace(/\\\r?\n/g, ' ')
    .match(/(?:\\[^\r\n]|[^\s\\])+/g);
if (!prerequisites?.length) throw new Error('Empty NVCC dependency manifest.');
const dependencies = [...new Set(prerequisites.map(file =>
    path.resolve(root, file.replace(/\\(.)/g, '$1').replace(/\$\$/g, '$'))))].sort();
const sourceHash = createHash('sha256');
sourceHash.update(JSON.stringify(dependencies));
for (const file of dependencies) {
    sourceHash.update(createHash('sha256').update(fs.readFileSync(file)).digest());
}
const fingerprint = JSON.stringify({ cccl, cuda, args: cubArgs, nvcc: nvccVersion.stdout, source: sourceHash.digest('hex') });
const stamp = path.join(root, 'build/cub_misc.build.json');
if (options['reuse-cub'] === '1') {
    if (!fs.existsSync(path.join(root, 'build/cub_misc.o')) || !fs.existsSync(stamp) || fs.readFileSync(stamp, 'utf8') !== fingerprint) {
        throw new Error('CUB object does not match this build. Rebuild without --reuse-cub 1.');
    }
} else {
    run(nvcc, cubArgs);
    fs.writeFileSync(stamp, fingerprint);
}
run(compiler, args);
run(compiler, [
    '-o', 'build/linux/x86_64/release/cuda_cub_misc',
    'build/cuda_cub_misc.o', 'build/cub_misc.o',
    '-L' + path.join(cuda, 'lib64'), '-Lbuild/linux/x86_64/release',
    '-Wl,-rpath,$ORIGIN', '-lluisa-dsl', '-lluisa-runtime',
    '-lluisa-vstl', '-lluisa-core', '-lluisa-ext-spdlog', '-lluisa-ext-lc-yyjson',
    '-lcuda', '-lcudart', '-ldl', '-luuid', '-lpthread', '-latomic',
]);
run(path.join(cuda, 'bin/cuobjdump'), ['--list-elf', 'build/cub_misc.o']);
