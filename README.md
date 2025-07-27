# Geometry Processing Demo

## Build Instructions

```bash
mkdir build
cd build
cmake ..
make
```

All executables will be generated under `build/bin`.  
Each `.cpp` file in the `src` directory will be built as a separate executable.

For example(if still under build folder):  
`bin/_decimation_qem`

## Dependencies

- [Eigen](https://eigen.tuxfamily.org/)
- [libigl](https://libigl.github.io/)

## License

This project uses [libigl](https://github.com/libigl/libigl), which is licensed under the [Mozilla Public License v2.0](https://www.mozilla.org/en-US/MPL/2.0/). See [`LICENSE`](./LICENSE) for details.

All original parts of this project are released under the MIT License. See [`LICENSE-MIT`](./LICENSE-MIT).