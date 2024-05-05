set -ex

path=./exputils/cpp
filename=$path/calc_dot

echo "now compiling..."
/usr/bin/g++-10 -pg -g -fdiagnostics-color=always $filename.cpp -o $filename -std=c++17 -Wall -Wextra -O3 -lz

echo "now running..."
$filename
gprof $filename gmon.out > gprof.profile
rm gmon.out