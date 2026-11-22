## Requirements

- Rust (stable, edition 2021)
- Target: `wasm32-unknown-unknown`
- Build tool: [Trunk](https://trunkrs.dev/)
- Runtime: Since wgpu is used, a modern browser with WebGPU support (e.g. Chrome or Edge) is recommended

Setup example:

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk
trunk build   # build
trunk serve   # start a local server
```

> Targets other than `wasm32-unknown-unknown` are not supported.

## References

Resources used in this project:

### Code and Examples

- [wgpu examples](https://github.com/gfx-rs/wgpu/tree/trunk/examples/src) - Reference implementations for rendering with wgpu
- [Trunk wasm_threads example](https://github.com/trunk-rs/trunk/tree/main/examples/wasm_threads) - Building WebAssembly projects with Trunk
- [lil-gui Kitchen Sink](https://lil-gui.georgealways.com/examples/kitchen-sink/) - Usage examples for the lil-gui controls
- [Rust/wgpuをWebブラウザで動かす (Zenn)](https://zenn.dev/matcha_choco010/articles/2022-07-11-wgpu-web) - Running wgpu in the browser

### Assets and Data

- [McGuire Computer Graphics Archive](https://casual-effects.com/data/) - Test models for computer graphics
- [Poly Haven](https://polyhaven.com/) - 3D models, HDRIs, and textures (CC0)
- [Benedikt Bitterli Rendering Resources](https://benedikt-bitterli.me/resources/) - Cornell Box scene

Assets from Poly Haven are released under CC0.

### Resource directories

- `resource/public/`: resources approved for distribution in the public repository
- `resource/private/`: local-only resources that must not be committed

Only `resource/public/` is copied into the browser build. Resources in `resource/private/` are ignored by Git. Asset-specific license notices are kept beside each public resource.

### Debug resources

Public builds use `trunk build` or `trunk build --release` with `Trunk.toml`.

Private debug resources are available only with:

```sh
trunk serve --config Trunk.debug.toml
```

### Cornell Box

The `resource/public/gltf/cornell_box` scene is a PBRT-to-glTF conversion of the Cornell Box from [Benedikt Bitterli Rendering Resources](https://benedikt-bitterli.me/resources/).

- Source scene: PBRT v4 version
- Files: `cornell_box_extended.gltf` (used by the renderer), `cornell_box_core.gltf`, `cornell_box.bin`
- License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- License notice: [`resource/public/gltf/cornell_box/LICENSE.txt`](resource/public/gltf/cornell_box/LICENSE.txt)

The source scene is stated by its original distributor to be in the public domain under CC0. CC0 allows copying, modification, commercial use, and redistribution without permission or attribution. Attribution is encouraged. The scene is therefore suitable for publication in this public repository. The license notice must remain with redistributed copies.
