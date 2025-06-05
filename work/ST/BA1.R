library("Seurat")
library("ggplot2")
library("cowplot")
library("dplyr")
library("hdf5r")

setwd("D:/R/002/ST/data/BA1/place")




library(Seurat)
library(hdf5r)
mydata = Load10X_Spatial(data.dir = "D:/R/002/ST/data/BA1/CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix",
                      filename = "CytAssist_FFPE_Mouse_Brain_Rep1_filtered_feature_bc_matrix.h5",
                      assay = "Spatial", 
                      slice = "Mouse Brain Coronal Section 1 (FFPE)",
                      filter.matrix = T)

head(mydata@meta.data,2)
##
plot1<-VlnPlot(mydata,features = "nCount_Spatial",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "nCount_Spatial")+theme(legend.position = "right")
plot_grid(plot1,plot2)


pdf(file="F4B1.pdf",width = 8,height = 8)
plot_grid(plot2)
dev.off()


##
plot1<-VlnPlot(mydata,features = "nFeature_Spatial",pt.size = 0.1)+NoLegend()
plot2<-SpatialFeaturePlot(mydata,features = "nFeature_Spatial")+theme(legend.position = "right")
plot_grid(plot1,plot2)


pdf(file="F4B2.pdf",width = 8,height = 8)
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
##

pdf(file="F4D.pdf",width = 15,height = 15)
SpatialFeaturePlot(mydata,features = c("Kcnma1","Plp1","Ptgds","Ttr","Apoe"))
dev.off()




p1<-SpatialFeaturePlot(mydata,features = "Kcnma1",pt.size.factor = 1)
p2<-SpatialFeaturePlot(mydata,features = "Kcnma1",alpha = c(0.1,1))
plot_grid(p1,p2)


mydata <- RunPCA(mydata, assay = "SCT", verbose = FALSE)
mydata <- FindNeighbors(mydata, reduction = "pca", dims = 1:30)
mydata <- FindClusters(mydata, verbose = FALSE)
mydata <- RunUMAP(mydata, reduction = "pca", dims = 1:30)



p1 <- DimPlot(mydata, reduction = "umap", label = TRUE)
p2 <- SpatialDimPlot(mydata, label = TRUE, label.size = 3)
p1 + p2

SpatialDimPlot(mydata, cells.highlight = CellsByIdentities(object = mydata, idents = c(2, 1, 4, 3,
                                                                                     5, 8)), facet.highlight = TRUE, ncol = 3)






#

# 1. 寻找各个cluster的标记基因
markers <- FindAllMarkers(
  mydata,
  only.pos = TRUE,        # 只保留阳性表达的marker
  min.pct = 0.25,         # 在至少25%的细胞中表达
  logfc.threshold = 0.25  # 表达量差异的最小阈值
)

# 2. 查看每个cluster的前5个marker基因
top5_markers <- markers %>%
  group_by(cluster) %>%
  top_n(n = 5, wt = avg_log2FC)

# 打印表格查看
print(top5_markers, n = Inf)


# 点图展示marker表达模式
DotPlot(mydata, features = unique(top5_markers$gene)) + 
  RotatedAxis()

pdf(file="注释BA1.pdf",width = 25,height = 10)
DotPlot(mydata, features = unique(top5_markers$gene)) + 
  RotatedAxis()
dev.off()




# 3. 可视化重要marker（以Kcnma1为例）
p1 <- VlnPlot(mydata, features = "Kcnma1")
p2 <- FeaturePlot(mydata, features = "Kcnma1")
p3 <- SpatialFeaturePlot(mydata, features = "Kcnma1")
p1 | p2 | p3

# 4. 根据marker基因及空间结构进行注释
new.cluster.ids <- c(
  "0" = "CTX_L4/5",          
  "1" = "TH_1",        
  "2" = "CTX_L6",        
  "3" = "TH_2",    
  "4" = "CC" ,
  "5" = "AMY",          
  "6" = "CTX_L2/3",         
  "7" = "VLM",        
  "8" = "TH_3",
  "9" = "HIP_CA1",         
  "10" = "CTX_PIR",        
  "11" = "STR_CPm",        
  "12" = "VL",
  "13" = "HIP_CA2/3",          
  "14" = "CTX_L3",         
  "15" = "STR_CPI",        
  "16" = "HIP_DG",
  "17" = "LHb"
)

# 5. 将新注释添加到数据中
mydata <- RenameIdents(mydata, new.cluster.ids)

# 6. 可视化注释结果
p1 <- DimPlot(mydata, label = TRUE,pt.size = 2,    
              label.size = 5) + NoLegend()
p2 <- SpatialDimPlot(mydata, 
                     label = TRUE, 
                     label.size = 3,
                     repel = TRUE) +  # 调整空间标签大小
  theme(
    legend.text = element_text(size = 12),  # 图例标签字体大小
    legend.title = element_text(size = 14)  # 图例标题字体大小（可选）
  )
p1 + p2



pdf(file="F4f.pdf")
p1 
dev.off()



pdf(file="F4g.pdf")
p2
dev.off()