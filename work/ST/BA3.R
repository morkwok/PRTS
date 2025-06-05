library("Seurat")
library("ggplot2")
library("cowplot")
library("dplyr")
library("hdf5r")

setwd("D:/R/002/ST/data/BA3/place")




library(Seurat)
library(hdf5r)
mydata = Load10X_Spatial(data.dir = "D:/R/002/ST/data/BA3/Visium_FFPE_Mouse_Brain_spatial",
                         filename = "Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5",
                         assay = "Spatial", 
                         slice = "Mouse Brain Coronal Section 3 (FFPE)",
                         filter.matrix = T)

head(mydata@meta.data,2)
##
plot1<-VlnPlot(mydata,features = "nCount_Spatial",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "nCount_Spatial")+theme(legend.position = "right")
plot_grid(plot1,plot2)

pdf(file="S10B1.pdf",width = 8,height = 8)
plot_grid(plot2)
dev.off()

##
plot1<-VlnPlot(mydata,features = "nFeature_Spatial",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "nFeature_Spatial")+theme(legend.position = "right")
plot_grid(plot1,plot2)


pdf(file="S10B2.pdf",width = 8,height = 8)
plot_grid(plot2)
dev.off()


#线粒体
mydata[["percent.mt"]]<-PercentageFeatureSet(mydata,pattern = "^mt[-]")
plot1<-VlnPlot(mydata,features = "percent.mt",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "percent.mt")+theme(legend.position = "right")
plot_grid(plot1,plot2)


mydata<-subset(mydata,subset=nFeature_Spatial>200&nFeature_Spatial<10000&nCount_Spatial>1000&nCount_Spatial<100000)


##
plot1<-VlnPlot(mydata,features = "nCount_Spatial",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "nCount_Spatial")+theme(legend.position = "right")
plot_grid(plot1,plot2)


#归一化
mydata<-SCTransform(mydata,assay = "Spatial",verbose = FALSE)


#查看特定基因表达
SpatialFeaturePlot(mydata,features = c("Kcnma1","Plp1","Ptgds","Ttr","Apoe"))

pdf(file="S10D.pdf",width = 15,height = 15)
SpatialFeaturePlot(mydata,features = c("Kcnma1","Plp1","Ptgds","Ttr","Apoe"))
dev.off()
##
p1<-SpatialFeaturePlot(mydata,features = "Kcnma1",pt.size.factor = 1)
p2<-SpatialFeaturePlot(mydata,features = "Kcnma1",alpha = c(0.1,1))
plot_grid(p1,p2)

