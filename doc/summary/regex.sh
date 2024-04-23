# Run this file in the 'Git bash' which can be opend
# from + button in the upper left corner of VSCode.

set -ex

for file in `find doc/summary -name "*.tex"`;
do
    echo $file
    sed -i -e "s/、/，/g" $file
    sed -i -e "s/,/，/g" $file
    sed -i -e "s/。/．/g" $file
    sed -i -e "s/ *$//g" $file
done