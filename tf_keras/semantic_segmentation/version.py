__version__ = '0.1.0'

'''
#############ISUES/BUGS#############



#############IMPROVEMENTS#############

 - Plot a prediction map per epoch can help in training a new model
 
 - adding image name to the tfrecors and dataset help keeping the full metadata
 
 - Find images with the lowest IoU / highest error can help identifying the pattern of errors or outlier input data
 
 - Run a negative buffer on masks or create a border so the prediction improve when it comes to the adjacent objects
 
 - test hyper param tunning
 
 
 
 
 #############Resolved#############
 
 - tf.keras.callbacks.ModelCheckpoint ignores the monitor parameter and only consider val_loss for saving the checkpoints
 I need to change the callbacks in both the train and finetune code to have IoU as the metric to save the best model
 if this does not work, I need to save all of the models and finally choose the one with the highest IoU
 --> (04/11/2022) I was using the word mointor instead of monitor! 

'''