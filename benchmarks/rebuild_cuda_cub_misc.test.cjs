const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { test } = require('node:test');

const root = path.resolve(__dirname, '..');
const cuda = process.env.CUDA_PATH || '/usr/local/cuda';
const release = path.join(root, 'build/linux/x86_64/release');
const libraries = ['luisa-dsl', 'luisa-runtime', 'luisa-vstl', 'luisa-core',
    'luisa-ext-spdlog', 'luisa-ext-lc-yyjson'].map(name =>
    ['so', 'a'].map(extension => path.join(release, `lib${name}.${extension}`))
        .find(file => fs.existsSync(file)));
const available = fs.existsSync(path.join(cuda, 'bin/nvcc')) &&
    libraries.every(Boolean);

test('CUB reuse rejects transitive header changes without a version bump', {
    skip: available ? false : 'requires CUDA and the benchmark SDK libraries',
}, t => {
    const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'lcpp cub reuse-'));
    t.after(() => fs.rmSync(fixture, { recursive: true, force: true }));
    const write = (file, contents) => {
        const target = path.join(fixture, file);
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, contents);
    };
    write('benchmarks/rebuild_cuda_cub_misc.cjs',
        fs.readFileSync(path.join(__dirname, 'rebuild_cuda_cub_misc.cjs')));
    write('benchmarks/cub_misc.cu', '#include "cub_misc.h"\n#include <cub/fixture.cuh>\nint fixture_value() { return FIXTURE_VALUE; }\n');
    write('benchmarks/cub_misc.h', '#pragma once\n');
    write('benchmarks/cuda_cub_misc.cpp', 'int main() { return 0; }\n');
    write('src/lcpp/parallel_primitive.h', '#pragma once\n');
    write('cccl/cub/cub/version.cuh', '#define CUB_VERSION 300500\n');
    write('cccl/thrust/thrust/version.h', '#define THRUST_VERSION 300500\n');
    write('cccl/libcudacxx/include/cuda/stream_ref', '#pragma once\n');
    write('cccl/cub/cub/fixture.cuh', '#include <cub/detail/policy.cuh>\n');
    write('cccl/cub/cub/detail/policy.cuh', '#define FIXTURE_VALUE 7\n');
    write('build/compile_commands.json', JSON.stringify([{
        directory: fixture,
        file: 'tests/device_scan_test.cpp',
        arguments: [process.env.CXX || 'c++', '-c', 'tests/device_scan_test.cpp', '-o', 'build/unused.o'],
    }]));
    fs.mkdirSync(path.join(fixture, 'build/linux/x86_64/release'), { recursive: true });
    for (const library of libraries) {
        fs.symlinkSync(library,
            path.join(fixture, 'build/linux/x86_64/release', path.basename(library)));
    }
    const build = (reuse = false) => spawnSync(process.execPath, [
        path.join(fixture, 'benchmarks/rebuild_cuda_cub_misc.cjs'),
        '--cccl', 'cccl', '--cuda', cuda, '--reuse-cub', reuse ? '1' : '0',
    ], { cwd: fixture, encoding: 'utf8' });
    const succeeds = result => assert.equal(result.status, 0, result.stderr || result.error?.message);
    const rejectsStale = result => {
        assert.notEqual(result.status, 0, 'accepted a stale CUB object after a dependency changed');
        assert.match(result.stderr, /CUB object does not match this build/);
    };

    succeeds(build());
    const object = path.join(fixture, 'build/cub_misc.o');
    const original = fs.readFileSync(object);
    fs.utimesSync(object, new Date(1000), new Date(1000));
    succeeds(build(true));
    assert.deepEqual(fs.readFileSync(object), original);
    assert.equal(fs.statSync(object).mtimeMs, 1000, 'recompiled a matching cached object');

    const policy = path.join(fixture, 'cccl/cub/cub/detail/policy.cuh');
    const originalTime = fs.statSync(policy);
    write('cccl/cub/cub/detail/policy.cuh', '#define FIXTURE_VALUE 9\n');
    fs.utimesSync(policy, originalTime.atime, originalTime.mtime);
    rejectsStale(build(true));
    succeeds(build());
    assert.notDeepEqual(fs.readFileSync(object), original);
    succeeds(build(true));

    // A newly included dependency must also invalidate the recorded object.
    write('cccl/cub/cub/detail/extra.cuh', '#pragma once\n');
    write('cccl/cub/cub/detail/policy.cuh', '#include "extra.cuh"\n#define FIXTURE_VALUE 9\n');
    rejectsStale(build(true));
    succeeds(build());
    succeeds(build(true));

    fs.unlinkSync(path.join(fixture, 'cccl/cub/cub/detail/extra.cuh'));
    const missing = build(true);
    assert.notEqual(missing.status, 0, 'accepted an object with a missing dependency');
    assert.match(missing.stderr, /Cannot query CUB dependencies/);
});
