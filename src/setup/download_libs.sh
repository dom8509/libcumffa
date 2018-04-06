cd ~/opt

type wget >/dev/null 2>&1 || { echo >&2 "I require wget but it's not installed. Aborting."; exit 1; }

rm -fR downloads
mkdir downloads

cd downloads

wget https://www.openssl.org/source/openssl-1.0.1j.tar.gz
wget http://www.shoup.net/ntl/ntl-6.1.0.tar.gz
wget https://gmplib.org/download/gmp/gmp-6.0.0a.tar.bz2
wget http://gforge.inria.fr/frs/download.php/file/30873/gf2x-1.1.tar.gz
