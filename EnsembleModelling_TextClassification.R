##### ENSEMBLE MODELLING FOR TEXT CLASSIFICATION #####
######################################################

# LOADING NECESSARY LIBRARY
require(RTextTools)
require(caret)

# READING THE DATASET
Mydata <- read.csv(file.choose())
Mydata$Class <- as.factor(Mydata$Class)

# CREATING THE DOCUMENT-TERM MATRIX
doc_matrix <- create_matrix(Mydata$Trimmed_Text, language="english", removeNumbers=TRUE,
                            stemWords=TRUE, removeSparseTerms=.998)

# CREATING CONTAINER FOR MODELLING
container <- create_container(doc_matrix, Mydata$Class, trainSize=1:4500,
                              testSize=4501:5485, virgin=FALSE)

# TRAINING MODELS
SVM <- train_model(container,"SVM")
GLMNET <- train_model(container,"GLMNET")
MAXENT <- train_model(container,"MAXENT")
SLDA <- train_model(container,"SLDA")
BOOSTING <- train_model(container,"BOOSTING")
BAGGING <- train_model(container,"BAGGING")
RF <- train_model(container,"RF")
NNET <- train_model(container,"NNET")
TREE <- train_model(container,"TREE")


# CLASSIFYING DATA USING BUILT MODELS
SVM_CLASSIFY <- classify_model(container, SVM)
GLMNET_CLASSIFY <- classify_model(container, GLMNET)
MAXENT_CLASSIFY <- classify_model(container, MAXENT)
SLDA_CLASSIFY <- classify_model(container, SLDA)
BOOSTING_CLASSIFY <- classify_model(container, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container, BAGGING)
RF_CLASSIFY <- classify_model(container, RF)
NNET_CLASSIFY <- classify_model(container, NNET)
TREE_CLASSIFY <- classify_model(container, TREE)


# ANALYTICS ON MODELS
analytics <- create_analytics(container,
                              cbind(SVM_CLASSIFY, SLDA_CLASSIFY,
                                    BOOSTING_CLASSIFY, BAGGING_CLASSIFY,
                                    RF_CLASSIFY, GLMNET_CLASSIFY,
                                    NNET_CLASSIFY, TREE_CLASSIFY,
                                    MAXENT_CLASSIFY))
summary(analytics)


# CREATE THE data.frame FOR SUMMARIES
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
ens_summary <-analytics@ensemble_summary
doc_summary <- analytics@document_summary


# ENSEMBLE AGREEMENT ANALYSIS
create_ensembleSummary(analytics@document_summary)


# EVALUATING ACCURACY USING CROSS VALIDATION
SVM <- cross_validate(container, 4, "SVM")
GLMNET <- cross_validate(container, 4, "GLMNET")
MAXENT <- cross_validate(container, 4, "MAXENT")
SLDA <- cross_validate(container, 4, "SLDA")
BAGGING <- cross_validate(container, 4, "BAGGING")
BOOSTING <- cross_validate(container, 4, "BOOSTING")
RF <- cross_validate(container, 4, "RF")
NNET <- cross_validate(container, 4, "NNET")
TREE <- cross_validate(container, 4, "TREE")


# EXPORTING THE DATA FOR ANALYTICS SUMMARY OF VARIOUS MODEL IN THE ENSEMBLE
write.csv(analytics@document_summary, "DocumentSummary.csv")




##### PREDICTING USING ENSEMBLE AGREEMENT (VOTING MECHANISM) ON COMPLETE DATASET #####

# CREATING CONTAINER FOR COMPLETE DATASET WITH 5485 DATA POINTS
container_total <- create_container(doc_matrix, Mydata$Class,
                              testSize=1:5485, virgin=FALSE)

# PREDICTING CLASS USING ENSEMBLE AGREEMENT ON COMPLETE DATASET
SVM_CLASSIFY <- classify_model(container_total, SVM)
GLMNET_CLASSIFY <- classify_model(container_total, GLMNET)
MAXENT_CLASSIFY <- classify_model(container_total, MAXENT)
SLDA_CLASSIFY <- classify_model(container_total, SLDA)
BOOSTING_CLASSIFY <- classify_model(container_total, BOOSTING)
BAGGING_CLASSIFY <- classify_model(container_total, BAGGING)
RF_CLASSIFY <- classify_model(container_total, RF)
NNET_CLASSIFY <- classify_model(container_total, NNET)
TREE_CLASSIFY <- classify_model(container_total, TREE)

# ANALYTICS SUMMARY ON PREDICTION ON THE COMPLETE DATASET
analytics1 <- create_analytics(container_total,
                              cbind(SVM_CLASSIFY, SLDA_CLASSIFY,
                                    BOOSTING_CLASSIFY, BAGGING_CLASSIFY,
                                    RF_CLASSIFY, GLMNET_CLASSIFY,
                                    NNET_CLASSIFY, TREE_CLASSIFY,
                                    MAXENT_CLASSIFY))
summary(analytics1)

Mydata_finalPred <- analytics1@document_summary
Mydata_finalPred$Text <- Mydata$Trimmed_Text
Mydata_finalPred$Class <- Mydata$Class
Mydata_finalPred$Prediction <- Mydata_finalPred$PROBABILITY_CODE

vars <- c("Class","Text","Prediction")

# GETTING SUBMISSION FILE READY
Final_Submission <- Mydata_finalPred[vars]
Output_Just_Class_Predictions <- Final_Submission$Prediction

# WRITING OUTPUTS TO CSV
write.csv(Mydata_finalPred, "Total_Predicted_Summary_V2.csv")
write.csv(Final_Submission, "Final_Submission_Summary_V2.csv")
write.csv(Output_Just_Class_Predictions, "Output_Predictions_V2.csv")


# CALCULATING ACCURACY USING CAONFUSION MATRIX
require(caret)
confusionMatrix(Final_Submission$Class,Final_Submission$Prediction)


# CALCULATING ACCURACY USING LOKTRA'S STATED MECHANISM
# Accuracy_Scoring_Meachanism_LOKTRA = [(Correctly Classified - Incorrectly Classified)/Total] = 98.35 using Ensemble Agreement
## ACCURACY IS 98.35 ##
cm = as.matrix(table(Actual = Final_Submission$Class, Predicted = Final_Submission$Prediction))
Score <- (sum(diag(cm)) - (sum(cm)-sum(diag(cm))))/sum(cm)
Score
