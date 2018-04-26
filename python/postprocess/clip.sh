#!/bin/bash
#$ -V
#$ -l h_rt=24:00:00
#$ -N clip_6_or_10
#$ -j y

module load batch_landsat/v4

#input 1: raster to clip
input1=$1

#input 2: raster to use extent of
input2=$2

#output: clipped raster
output1=$3

#clipping shapefile
shapefile=$4


function gdal_extent() {
    if [ -z "$1" ]; then 
        echo "Missing arguments. Syntax:"
        echo "  gdal_extent <input_raster>"
        return
    fi
    EXTENT=$(gdalinfo $1 |\
        grep "Upper Left\|Lower Right" |\
        sed "s/Upper Left  //g;s/Lower Right //g;s/).*//g" |\
        tr "\n" " " |\
        sed 's/ *$//g' |\
        tr -d "[(,]")
    echo -n "$EXTENT"
}

function ogr_extent() {
    if [ -z "$1" ]; then 
        echo "Missing arguments. Syntax:"
        echo "  ogr_extent <input_vector>"
        return
    fi
    EXTENT=$(ogrinfo -al -so $1 |\
        grep Extent |\
        sed 's/Extent: //g' |\
        sed 's/(//g' |\
        sed 's/)//g' |\
        sed 's/ - /, /g')
    EXTENT=`echo $EXTENT | awk -F ',' '{print $1 " " $4 " " $3 " " $2}'`
    echo -n "$EXTENT"
}

img_ext=$(gdal_extent $input2)
shp_ext=$(ogr_extent $shapefile)

pix=$(gdalinfo $input2 \
    | grep "Pixel Size" | sed "s/Pixel.*(//g;s/,/ /g;s/)//g")
pix_sz="$pix $pix"
echo 'Pix size is:' $pix_sz

echo "Extent of stacked images and extent of shapefile:"
echo $img_ext
echo $shp_ext

new_ext=""

for i in 1 2 3 4; do
    # Get the ith coordinate from sequence
    r=$(echo $img_ext | awk -v i=$i '{ print $i }')
    v=$(echo $shp_ext | awk -v i=$i '{ print $i }')
    pix=$(echo $pix_sz | awk -v i=$i '{ print $i }')
    # Quick snippit of Python
    ext=$(python -c "\\
        offset=int(($r - $v) / $pix); \
        print $r - offset * $pix\
        ")
    echo $ext
    new_ext="$new_ext $ext"
done

echo "Calculated new extent:"
echo $new_ext

# Now, unfortunately, gdalwarp wants us to specify xmin ymin xmax ymax
# In this case, this corresponds to the upper left X, lower right Y, lower right
# X, and upper left Y
warp_ext=$(echo $new_ext | awk '{ print $1 " " $4 " " $3 " " $2 }')
echo "gdalwarp extent:"
echo $warp_ext

# Perform the clip:

    
gdalwarp -of GTiff -te $warp_ext -tr 30 30 \
    -cutline $shapefile \
    $input1 $output1

