library(TDA)
library(ggplot2)
library(TDA)
library(scatterplot3d)








generateBettiCurves <-
  function(diag, epsilonStepSize, numberOfEpsilons) {
    
    epsilon <- 0.0000000000
    epsStep <- epsilonStepSize
    numberOfFiltrationSeps <- numberOfEpsilons
    betti0 <-vector(mode = "numeric", length = numberOfFiltrationSeps)
    filtrationTime <-vector(mode = "numeric", length = numberOfFiltrationSeps)
    diag_object <- diag
    
    bettiNumberVector <- diag_object$diagram[,1]
    birthVector <- diag_object$diagram[,2]
    deathVector <- diag_object$diagram[,3]
    
    for (i in 1:numberOfFiltrationSeps) {
      
      filtrationTime[i] <- epsilon
      tempBettiNumberCounter <- 0
      for (j in 1:length(bettiNumberVector)) {
        
        if (bettiNumberVector[j] == 0) {
          
          if (epsilon >= birthVector[j] & epsilon < deathVector[j]) {
            
            tempBettiNumberCounter = tempBettiNumberCounter + 1
          }
        }
        
      }
      if (tempBettiNumberCounter == 0) {
        
        betti0[i] <- 1
      }
      else{
        
        betti0[i] <- tempBettiNumberCounter
        epsilon <- epsilon + epsStep
      }
    }
    
    
    
    return(list(filtrationTime, betti0))
    
  }


readDataProcessSave <- function(eStep=0.01, filt_time=200){
  eps <- eStep
  filtTime <- filt_time
files <- list.files(path = "C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\", pattern=".csv$")
tempBettiHold = vector(mode="numeric", length=100)
for (i in 1:length(files)){
  tempFile <- read.csv(paste("C:\\Users\\micha\\PycharmProjects\\betti\\cloud_data\\Class1\\",files[i], sep=""), header=FALSE)
  temp_matrix = data.matrix(tempFile, rownames.force = NA)
  
  Diag <-ripsDiag( X = temp_matrix, maxdimension, maxscale,library = "GUDHI", printProgress = FALSE)
  tempBetti <- generateBettiCurves(Diag,eps,filtTime)
  tempBettiHold <- data.matrix(cbind(tempBettiHold, tempBetti[[2]]))

}

tempBettiHold <- t(tempBettiHold[,-1])
bettiDF <- data.frame(tempBettiHold)

write.csv(bettiDF, file="bettiCurvesClass2New1.csv", row.names = FALSE)
  
}



readDataProcessSaveFromPython <- function(eStep=0.01, filt_time=200, path_to_data, path_and_file_name_to_save){
  eps <- eStep
  filtTime <- filt_time
  maxdimension <-1
  maxscale <-5
  files <- list.files(path = path_to_data, pattern=".csv$")
  tempBettiHold = vector(mode="numeric", length=100)
  for (i in 1:length(files)){
    tempFile <- read.csv(paste(path_to_data,files[i], sep=""), header=FALSE)
    temp_matrix = data.matrix(tempFile, rownames.force = NA)
    
    Diag <-ripsDiag( X = temp_matrix, maxdimension, maxscale,library = "GUDHI", printProgress = FALSE)
    tempBetti <- generateBettiCurves(Diag,eps,filtTime)
    tempBettiHold <- data.matrix(cbind(tempBettiHold, tempBetti[[2]]))
    
  }
  
  tempBettiHold <- t(tempBettiHold[,-1])
  bettiDF <- data.frame(tempBettiHold)
  
  write.csv(bettiDF, file=path_and_file_name_to_save, row.names = FALSE)
  
}


myArgs <- commandArgs(trailingOnly = TRUE)

arg1 <- myArgs[1]
arg2 <- myArgs[2]

readDataProcessSaveFromPython(0.01, 200,arg1, arg2)

cat(myArgs[2])