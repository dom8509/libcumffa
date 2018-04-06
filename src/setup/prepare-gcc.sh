rm -fR mpc-1.0.1
rm -fR gmp-6.0.0
rm -fR mpfr-3.1.2
rm -fR gcc-4.8.3
rm -fr binutils-2.24
rm -fR build
rm -fR $HOME/opt

tar -xf gcc-4.8.3.tar.bz2
tar -xf binutils-2.23.tar.gz
tar -xf mpc-1.0.1.tar.gz
tar -xf gmp-6.0.0a.tar
tar -xf mpfr-3.1.2.tar.bz2

mv mpc-1.0.1 gcc-4.8.3/mpc
mv gmp-6.0.0 gcc-4.8.3/gmp
mv mpfr-3.1.2 gcc-4.8.3/mpfr

mkdir build
mkdir build/gcc
mkdir build/binutils

mkdir $HOME/opt
mkdir $HOME/opt/gcc-4.8.3

export PREFIX=$HOME/opt/gcc-4.8.3

#cd build/binutils
#../../binutils-2.23/configure --prefix="$PREFIX" --disable-nls
#make check
#make install

cd build/gcc
unset LIBRARY_PATH CPATH C_INCLUDE_PATH PKG_CONFIG_PATH CPLUS_INCLUDE_PATH INCLUDE
../../gcc-4.8.3/configure --prefix="$PREFIX" --disable-nls --enable-languages=c,c++ --disable-multilib
make
make install