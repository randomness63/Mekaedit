package meka.classifiers.multilabel;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import meka.classifiers.multilabel.BR;
import meka.classifiers.multilabel.MULANedit;
import meka.classifiers.multilabel.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import meka.core.MLEvalUtils;
import meka.core.MLUtils;
import meka.core.Result;
import weka.core.Utils;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;

public class MultiRun {
    public static void main(String[] args)
    {
        try
        {
            J48 baseClassifier_J48=new J48();
            Evaluation evaluator=new Evaluation();
            Result result = new Result();
            System.out.println("Load the training data");
    		Instances trainingData = Evaluation.loadDataset(args);

            String testingDataFilename = Utils.getOption("-T", args);
            String modelFilename = Utils.getOption("model", args);
            
            //Setup
            
            BR h = new BR();
            h.setClassifier(baseClassifier_J48);
            System.out.println("Build the model");
            h.buildClassifier(trainingData);
            
            System.out.println("Save the model");
            SerializationHelper.write(modelFilename, h);
            
            System.out.println("Load the model");
            BR learner2;
            learner2 = (BR) SerializationHelper.read(modelFilename);
            
			Instances D_test = null;

            if(Utils.getOptionPos('T',args) >= 0) {
				// load separate test set
				try {
					D_test = Evaluation.loadDataset(args,'T');
					MLUtils.prepareData(D_test);
				} catch(Exception e) {
					throw new Exception("[Error] Failed to Load Test Instances from file.", e);
				}
			}
            
            result = evaluator.evaluateModel(learner2,trainingData,D_test,"1");
            
            System.out.println(result.toString());
        }
        catch (Exception ex)
        {
            Logger.getLogger(MultiRun.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

}
