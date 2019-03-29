package mulanedit;
import mulan.classifier.neural.model.*;
import java.util.*;

public interface ExtendedNeuralNet extends NeuralNet{

    void setDropoutMatrix(ArrayList<ArrayList<Double>> dMatrix);
    
    double[] feedForward(double[] inputPattern, boolean dropout);

}