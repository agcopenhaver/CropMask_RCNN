library(raster)
library(rgdal)
files <- list.files(path="/home/rave/deeplearn_imagery/data/raw/stephtest/PROJECTED_IMAGES/", pattern="*wgs$", full.names=T, recursive=FALSE)
print(files)
rasters <- lapply(files, raster)
#resources
# https://gis.stackexchange.com/questions/187798/create-polygons-of-the-extents-of-a-given-raster-in-r
# https://www.r-bloggers.com/getting-rasters-into-shape-from-r/
# make all values the same. 
for (i in 1:length(rasters)) {
  rasters[[i]] <- rasters[[i]] > -Inf
  }

gdal_polygonizeR <- function(x, outshape=NULL, gdalformat = 'ESRI Shapefile', 
                             pypath=NULL, readpoly=TRUE, quiet=TRUE) {
  if (isTRUE(readpoly)) require(rgdal)
  if (is.null(pypath)) {
    pypath <- Sys.which('gdal_polygonize.py')
  }
  if (!file.exists(pypath)) stop("Can't find gdal_polygonize.py on your system.") 
  owd <- getwd()
  on.exit(setwd(owd))
  setwd(dirname(pypath))
  if (!is.null(outshape)) {
    outshape <- sub('\\.shp$', '', outshape)
    f.exists <- file.exists(paste(outshape, c('shp', 'shx', 'dbf'), sep='.'))
    if (any(f.exists)) 
      stop(sprintf('File already exists: %s', 
                   toString(paste(outshape, c('shp', 'shx', 'dbf'), 
                                  sep='.')[f.exists])), call.=FALSE)
  } else outshape <- tempfile()
  if (is(x, 'Raster')) {
    require(raster)
    writeRaster(x, {f <- tempfile(fileext='.asc')})
    rastpath <- normalizePath(f)
  } else if (is.character(x)) {
    rastpath <- normalizePath(x)
  } else stop('x must be a file path (character string), or a Raster object.')
  system2('python', args=(sprintf('"%1$s" "%2$s" -f "%3$s" "%4$s.shp"', 
                                  pypath, rastpath, gdalformat, outshape)))
  if (isTRUE(readpoly)) {
    shp <- readOGR(dirname(outshape), layer = basename(outshape), verbose=!quiet)
    return(shp) 
  }
  return(NULL)
}

polys <- lapply(rasters, gdal_polygonizeR)

for (i in 1:length(polys)) {
  writeOGR(obj=polys[[i]], dsn="/home/rave/deeplearn_imagery/data/raw/stephtest/wv2masks/", layer=toString(i), driver="ESRI Shapefile") # this is in geographical projection
}
