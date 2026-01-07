# manga-dev

## How to install the Anaconda development environment (Windows)

1.  Create manga-dev environment:

    ```bash
    conda create -c conda-forge -n manga-dev python=3.12
    conda activate manga-dev
    ```

2. Install the Astropy packages:

   Use conda to install packages.
    ```bash
    conda install -c conda-forge nomkl numpy scipy lmfit
    conda install -c conda-forge pytensor pymc arviz nutpie dm-tree
    conda install -c conda-forge jax jaxlib
    conda install -c conda-forge \
                    pandas pytz h5py pyarrow fsspec s3fs bottleneck \
                    certifi tqdm mpmath jplephem \
                    beautifulsoup4 html5lib bleach
    conda install -c conda-forge ipython jupyter dask
    conda install -c conda-forge astropy astroquery reproject asdf-astropy
    conda install -c conda-forge pvextractor
    conda install -c conda-forge matplotlib seaborn xarray-einstats numba
    conda install -c conda-forge m2w64-toolchain libpython
    ```

    Some packages need used pip to install.
    ```
    pip install ppxf spectral_cube sdss-access
    ```