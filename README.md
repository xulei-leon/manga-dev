# manga-dev

## How to install the Anaconda development environment

1.  Create manga-dev environment:

    ```bash
    conda create -c conda-forge -n manga-dev python=3.12
    conda activate manga-dev
    ```

2. Install the Astropy packages:

   Use conda to install packages.
    ```bash
    conda install -c conda-forge astropy
    conda install -c conda-forge scipy matplotlib
    conda install -c conda-forge ipython jupyter dask h5py pyarrow beautifulsoup4 html5lib bleach pandas pytz jplephem mpmath asdf-astropy bottleneck fsspec s3fs certifi tqdm
    conda install -c conda-forge pvextractor
    conda install -c conda-forge astroquery reproject
    conda install -c conda-forge emcee
    conda install -c conda-forge pymc jax jaxlib arviz
    ```

    Some packages need used pip to install.
    ```
    pip install ppxf spectral_cube sdss-access
    ```