#installes gmp, gf2x and ntl

cd ~/opt/downloads

rm -fR ntl-6.1.0
rm -fR gfx-1.1
rm -fR gmp-6.0.0

tar -xf ntl-6.1.0.tar.*
tar -xf gf2x-1.1.tar.*
tar -xf gmp-6.0.0a.tar.*

cd gmp-6.0.0
./configure --prefix=$HOME/opt ABI=64 CFLAGS="-m64 -O2" --enable-cxx=yes
make -j45 clean
make -j45
make -j45 check
make -j45 install

cd ../gf2x-1.1	
./configure --prefix=$HOME/opt ABI=64 CFLAGS="-m64 -O2"
make -j45 clean
make -j45
make -j45 check
make -j45 install

cd ../ntl-6.1.0/src
./configure DEF_PREFIX=$HOME/opt NTL_GMP_LIP=on NTL_GF2X_LIB=on CFLAGS="-m64 -O2"
make -j45 clean
make -j45
make -j45 check
make -j45 install

#openssl:
#export CCACHE_DISABLE=1
#export CFLAGS=-fPIC
#./config shared --prefix=/your/path 