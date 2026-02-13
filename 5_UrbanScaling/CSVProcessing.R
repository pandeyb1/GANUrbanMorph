dir = "/Users/9oy/Documents/Data/IM3/Buildings/MAv1"

library(sf)
library(data.table)

setwd(dir)
files = list.files(".",pattern="*.csv")

processfile = function(file){
  data = as.data.frame(fread(file,header = T,drop=1))
  cen = strsplit(data$Centroid,"/")
  cen = do.call(rbind.data.frame, cen)
  colnames(cen) = c("lat","lon")
  
  polygons = strsplit(data$Footprint2D, "_") %>%
    lapply(function(x) {
      coords = strsplit(x, "/")
      coords = matrix(as.numeric(unlist(coords)), ncol = 2, byrow = TRUE)
      coords = cbind(coords[,2],coords[,1])
      nump = dim(coords)
      coords = rbind(coords,coords[1,])
      st_polygon(list(coords))
    })
  obj =  st_sf(geometry = st_sfc(polygons),crs="EPSG:4326")
  obj$lat = cen$lat
  obj$lon = cen$lon
  data= data[,c("ID","State_Abbr","Area","Area2D","Height","NumFloors","WWR_surfaces","BuildingType","Standard")]
  obj = cbind(obj,data)
  return(obj)
}

for(i in 1:length(files)){
  if(i == 1){
    fname = files[i]
    stname = unlist(strsplit(fname,".csv"))
    outdata = processfile(fname)
    st_write(outdata,"buildingspoly.gpkg", stname)
#    st_write(outdata,paste("buildingspoly","_",stname,".gpkg",sep=""))
  }else{
    fname = files[i]
    stname = unlist(strsplit(fname,".csv"))
    outdata = processfile(fname)
    st_write(outdata,"buildingspoly.gpkg", stname,append=T)
#    st_write(outdata,paste("buildingspoly","_",stname,".gpkg",sep=""))
  }
  print(i)
  flush.console()
}