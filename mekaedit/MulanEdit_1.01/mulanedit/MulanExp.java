package mulanedit;
import java.io.FileReader;
import java.util.List;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.neural.BPMLL;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.Evaluation;

public class MulanExp {

    public static void main(String[] args) throws Exception {
    	
        String arffFilename = Utils.getOption("arff", args);
        
        System.out.println(arffFilename);
        String xmlFilename = Utils.getOption("xml", args);
        double l2reg = Double.valueOf(Utils.getOption("l", args));
        String dropout = Utils.getOption("d", args);

        MultiLabelInstances trainset = new MultiLabelInstances(arffFilename, xmlFilename);
        BPMLLwD model = new BPMLLwD();
        model.setLearningRate(0.01);
        model.setHiddenLayers(new int[]{30});
        model.setTrainingEpochs(100);
        model.setWeightsDecayRegularization(l2reg);
        
        System.out.println("Set up done");
        
        model.build(trainset);
        
        /* String unlabeledFilename = Utils.getOption("unlabeled", args);
        FileReader reader = new FileReader(unlabeledFilename);
        Instances unlabeledData = new Instances(reader);

        int numInstances = unlabeledData.numInstances();
        

        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = unlabeledData.instance(instanceIndex);
            MultiLabelOutput output = model.makePrediction(instance);
            System.out.println(instance);
            System.out.println(output);
        }
      
        */
        Evaluator eval = new Evaluator();
        MultipleEvaluation results2;
        results2 = eval.crossValidate(model, trainset, 10);
        System.out.println(results2);
        
    }
}