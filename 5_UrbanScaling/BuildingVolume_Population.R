rm(list=ls())
gc()

dir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/MetroBuildings"
library(sf)
library(data.table)
library(tidyverse)
library(ggplot2)
library(scales)
library(ggrepel)
library(gridExtra)
library(raster)
setwd(dir)
set.seed(1230)

# 381 MSAs excluding PR
msa = read_sf("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/nhgis0039_shape/nhgis0039_shapefile_tl2015_us_cbsa_2015/US_cbsa_2015.shp")
msam1 = msa %>% filter(LSAD=="M1") ## 388 METROS
orignames = msam1$NAME
metronames = as.character(msam1$NAME)
stnames = gsub(" ","",sapply(strsplit(metronames,","),"[[",2))
msam1 = msam1[which(!(stnames %in% "PR")),]
selnames = msam1$NAME
dim(msam1)
outfile ="/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/MSA_notPR.gpkg"
if(!file.exists(outfile)){
write_sf(msam1,"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/MSA_notPR.gpkg")
}
colnames(msam1)
# pop
pop = read.csv("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/nhgis0039_csv/nhgis0039_ds214_2015_cbsa.csv")
pop = pop[,c("GISJOIN","ACK2E001")]

msam1 = merge(msam1,pop,by=c("GISJOIN"),all.x=T)
bvolfile = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/BuildingVolume2015MSA.gpkg"

if(!file.exists(bvolfile)){

files = list.files(pattern="*.gpkg") ## These were generated using exportmsalevelbuildingsdata.py
print(files)
vols = rep(NA,length(files))
NAME = rep(NA,length(files))

for(i in 1: length(files)){
  NAME[i] = strsplit(files[i],".gpkg")[[1]][1]
  if(NAME[i] %in% c("Coeur d'Alene, ID")){
    shp = read_sf(files[i])
  }else{    query = paste("SELECT volume FROM ","'", NAME[i],"'",sep="")
    shp = read_sf(files[i],query=query)

  }
  vols[i] = sum(shp$Volume,na.rm=T)
  print(i)
  flush.console()
}
NAME[NAME=="Louisville_Jefferson County, KY-IN.gpkg"] = "Louisville/Jefferson County, KY-IN"
bvol = data.frame(NAME,vols)
bvols = merge(msam1,bvol,by="NAME")
write_sf(bvols,bvolfile)
}else{
  bvols = read_sf(bvolfile)
}

bvols = bvols[(bvols$vols>0),]

# Beta
mod = lm(log(vols)~log(ACK2E001),data=bvols)
summary(mod)
round(confint(mod,level=0.95),3)

# Load libraries
library("lmtest")
library("sandwich")
mod1 = coeftest(mod, vcov = vcovHC(mod, type = "HC0"))
round(confint(mod1,level=0.95),2)
label_text <- paste0(
  "beta == ", round(as.numeric(coef(mod)[2]),2), " * ' (95% CI: ", 
  round(confint(mod1,level=0.95),2)[2,1], 
  " - ", round(confint(mod1,level=0.95),2)[2,2], ")'"
)
# Plot
x = 10^c(4.5,5,6,7,8)
slope_low = confint(mod,level=0.95)[2,1]
slope_high = confint(mod,level=0.95)[2,2]
intercept_low = exp(confint(mod,level=0.95)[1,1])
intercept_high = exp(confint(mod,level=0.95)[1,2])
slope_mid  = as.numeric(coef(mod)[2])
intercept_mid = exp(as.numeric(coef(mod)[1]))
y_low= x^slope_low * intercept_low
y_high= x^slope_high * intercept_high
y_mid = x^slope_mid * intercept_mid
confdat = data.frame(x,y_low,y_high,y_mid)


p1 = ggplot(data=bvols %>% as.data.frame(.),aes(x=ACK2E001,y=vols)) +
  geom_point(alpha=0.15,size=3) + geom_smooth(method="lm",se=F,col="red") + theme_bw()+
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) + xlab("Population") + ylab(expression("Building Volume " * (m^3))) +
  theme(axis.title = element_text(size=16),axis.text = element_text(size=13)) +# geom_text_repel() +
annotate(geom="text", x=10^6.5, y=10^7, 
         label=label_text, parse=T,
         color="black",size=5) +
  geom_line(data=confdat,aes(x =x,y=y_low),linetype="dashed") + 
  geom_line(data=confdat,aes(x=x,y=y_high),linetype="dashed")

UA_2010 = read_sf("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/nhgis0040_shape/nhgis0040_shapefile_tl2010_us_urb_area_2010/US_urb_area_2010.shp")
UA_2010 = UA_2010[grepl("Los Angeles-",UA_2010$NAME10),"NAME10"]
LS2010 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/Landscan/landscan-global-2010-assets/landscan-global-2010.tif"
LS2020 = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/Data/Landscan/landscan-global-2020-assets/landscan-global-2020.tif"
LS2010 = raster(LS2010)
LS2020 = raster(LS2020)
UA_2010 = st_transform(UA_2010,crs(LS2010))

LS2010 = mask(crop(LS2010,UA_2010),UA_2010)
LS2020 = mask(crop(LS2020,UA_2010),UA_2010)

pop2010 = cellStats(LS2010,sum)
pop2020 = cellStats(LS2020,sum)
bv2010 = pop2010^slope_mid * intercept_mid
bv2020 = pop2020^slope_mid * intercept_mid

# Plot
x = c(pop2010,pop2020)
slope_low = confint(mod,level=0.95)[2,1]
slope_high = confint(mod,level=0.95)[2,2]
intercept_low = exp(confint(mod,level=0.95)[1,1])
intercept_high = exp(confint(mod,level=0.95)[1,2])
slope_mid  = as.numeric(coef(mod)[2])
intercept_mid = exp(as.numeric(coef(mod)[1]))
y_low = x^slope_low * intercept_low
y_high = x^slope_high * intercept_high
y_mid = x^slope_mid * intercept_mid
confdat = data.frame(x,y_low,y_high,y_mid)
(7375402746 - 7290371895)*100/7290371895

GANBF = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF2020_GAN.tif")
GANBH = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH2020_GAN.tif")
BF2010 = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF2010.tif")
BH2010 = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH2010.tif")
BF2020 = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BF_2020.tif")
BH2020 = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/BH_2020.tif")
cmaskdat = raster("/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/tmp/ChangeMask2010_2020.tif")
GANBH[GANBH > 75] =75
BH2010[BH2010 > 75] =75
BH2020[BH2020 > 75] =75
GANBV = GANBF * GANBH * 30*30
BV2010 = BF2010 * BH2010 * 30*30
BV2020 = BF2020 * BH2020* 30*30

UA_2010p = st_transform(UA_2010,crs(GANBF))
GANBV = mask(crop(GANBV,UA_2010p),UA_2010p)
BV2010 = mask(crop(BV2010,UA_2010p),UA_2010p)
BV2020 = mask(crop(BV2020,UA_2010p),UA_2010p)

BV2010e = cellStats(BV2010,sum)
GANBVe = cellStats(BV2010,sum) + cellStats(GANBV * cmaskdat,sum)
BV2020e = cellStats(BV2010,sum) + cellStats(BV2020 * cmaskdat,sum)
(GANBVe - BV2010e)*100 / BV2010e
(BV2020e - BV2010e)*100 / BV2010e
(y_mid[2] - y_mid[1])*100/y_mid[1]
(y_low[2] - y_low[1])*100/y_low[1]
(y_high[2] - y_high[1])*100/y_high[1]


(BV2010e - y_mid[1]) * 100 / y_mid[1]
(y_high[1] - y_mid[1]) * 100 / y_mid[1]

p1 = p1 + geom_point(aes(x = pop2010, y = BV2020e), color = "red", size = 4, shape = 20) #+ 
#  geom_point(aes(x = pop2020, y = BV2020e), color = "red", size = 4, shape = 20)
  
p2 = ggplot()+ 
  geom_point(aes(x = pop2010, y = BV2010e), color = "red", size = 4, shape = 20) + 
  geom_point(aes(x = pop2020, y = BV2020e), color = "red", size = 4, shape = 20) + 
  geom_point(aes(x = pop2020, y = GANBVe), color = "blue", size = 4, shape = 20) +
  geom_segment(aes(x = pop2010, y = BV2010e, xend = pop2020, yend = BV2020e),
               arrow = arrow(),col="red") + 
  geom_segment(aes(x = pop2010, y = BV2010e, xend = pop2020, yend = GANBVe),
               arrow = arrow(),col="blue") + 
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) + xlab("Population") + 
  ylab("") +
  theme_bw() + 
  theme(axis.title = element_text(size=16),axis.text = element_text(size=12)) + 
  annotate("text",x = pop2010+5000, y = BV2020e + 99000000,label="2010") + 
  annotate("text",x = pop2020-5000, y = BV2020e + 99000000,label="2020")
  
outplt = grid.arrange(p1,p2,ncol=2)

ggsave("/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/5_UrbanScaling/ScalingOutput.jpg",
       outplt,
  width=8.8,height=3.333)
