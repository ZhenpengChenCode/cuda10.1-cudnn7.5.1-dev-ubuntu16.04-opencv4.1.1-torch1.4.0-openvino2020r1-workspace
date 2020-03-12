FROM ubuntu:16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# cuda10.1
RUN apt-get update && apt-get install -y --no-install-recommends \
ca-certificates apt-transport-https gnupg-curl && \
    NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --auto-remove -y gnupg-curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-1 && \
ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411"

# cudnn7.6.5
ENV CUDNN_VERSION 7.5.1
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Intel CPU Options
ARG AVX=ON
ARG AVX2=ON
ARG SSE41=ON
ARG SSE42=ON
ARG SSSE3=ON
ARG TBB=ON

# Include Intel IPP support
# Intel IPP software building blocks are highly optimized instruction sets (using Intel AVX, AVX2 and SSE).It offers a special subset of functions for image processing and computer vision called the IPP-ICV libraries. More information can be found here. Also here you can find some information about speedup.
ARG IPP=ON

# Include NVidia Cuda Runtime support
ARG CUDA=OFF

# Include NVidia Cuda Fast Fourier Transform (FFT) library support
ARG CUFFT=OFF

# Include NVidia Cuda Basic Linear Algebra Subprograms (BLAS) library support
ARG CUBLAS=OFF

# Include OpenCL Runtime support
ARG OPENCL=OFF

# Include OpenCL Shared Virtual Memory support" OFF ) experimental
ARG OPENCL_SVM=OFF

ARG OPENGL=ON

ARG GSTREAMER=ON

ARG FFMPEG=ON
ARG GTK=OFF
ARG QT=OFF
ARG NONFREE=OFF

# Include Intel Perceptual Computing SDK
ARG INTELPERC=OFF

ARG PREFIX=/usr/local
ARG VERSION=3.3.0

ARG PYTHON_BIN=/usr/local/bin/python
ARG PYTHON_LIB=/usr/local/lib/libpython3.so

WORKDIR /

# LAPACKE is the C wrapper for the standard F90 LAPACK library. Honestly, its
# easier (and more efficient) to do things directly with LAPACK just as long as
# you store things column-major. LAPACKE ends up calling (in some fashion) the
# LAPACK routines anyways.
RUN apt-get update -q -y && apt-get install -y \
        build-essential \
        cmake \
        yasm \
        pkg-config \
        libswscale-dev \
        libeigen3-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libjasper-dev \
        libavformat-dev \
        libpq-dev \
        libboost-all-dev \
        libgstreamer1.0-0 libgstreamer1.0-dev gstreamer1.0-libav gstreamer1.0-plugins-base \
        libblas-dev \
        liblapacke liblapacke-dev \
        libopenblas-dev libopenblas-base \
        libatlas-dev libatlas-base-dev \
        liblapacke-dev liblapacke \
        && dpkg-query -Wf '${Installed-Size}\t${Package}\n' | sort -n \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists

RUN pip install --upgrade pip \
 && pip install numpy \
 && pip install scipy \
 && rm -rf ~/.cache/pip

RUN curl --silent --location --location-trusted \
        --remote-name https://github.com/opencv/opencv/archive/$VERSION.tar.gz \
    && tar xf $VERSION.tar.gz -C / \
    && mkdir /opencv-$VERSION/cmake_binary \
    && cd /opencv-$VERSION/cmake_binary \
    && cmake \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_INSTALL_PREFIX=$PREFIX \
        -DOPENCV_ENABLE_NONFREE=$NONFREE \
        -DBUILD_opencv_java=OFF \
        -DWITH_CUDA=$CUDA \
        -DWITH_CUBLAS=$CUBLAS \
        -DWITH_CUFFT=$CUFFT \
        -DENABLE_AVX=$AVX \
        -DENABLE_AVX2=$AVX2 \
        -DENABLE_SSE41=$SSE41 \
        -DENABLE_SSE42=$SSE42 \
        -DENABLE_SSSE3=$SSSE3 \
        -DWITH_OPENGL=$OPENGL \
        -DWITH_GTK=$GTK \
        -DWITH_GSTREAMER=$GSTREAMER \
        -DWITH_OPENCL=$OPENCL \
        -DWITH_OPENCL_SVM=$OPENCL_SVM \
        -DWITH_TBB=$TBB \
        -DWITH_JPEG=ON \
        -DWITH_WEBP=ON \
        -DWITH_TIFF=ON \
        -DWITH_PNG=ON \
        -DWITH_QT=$QT \
        -DWITH_IPP=$IPP \
        -DWITH_EIGEN=ON \
        -DWITH_V4L=ON \
        -DWITH_INTELPERC=$INTELPERC \
        -DWITH_FFMPEG=$FFMPEG \
        -DENABLE_PRECOMPILED_HEADERS=ON \
        -DBUILD_opencv_python2=NO \
        -DBUILD_opencv_python3=ON \
        -DPYTHON3_EXECUTABLE=$PYTHON_BIN \
        -DPYTHON3_LIBRARIES=$PYTHON_LIB \
        -DPYTHON_LIBRARIES=$PYTHON_LIB \
        -DPYTHON3_INCLUDE_DIR=$($PYTHON_BIN -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON3_PACKAGES_PATH=$($PYTHON_BIN -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
        -DBUILD_DOCS=NO \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF \
    && make install \
    && rm -rf /$VERSION.tar.gz /opencv-$VERSION
