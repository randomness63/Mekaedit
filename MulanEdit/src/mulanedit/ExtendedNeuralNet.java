package mulanedit;
import mulan.classifier.neural.model.*;
import java.util.*;

public interface ExtendedNeuralNet extends NeuralNet{
	//extends Neural net to provide 
	
	/**
     * Sets the Array List of Array lists for the DropoutMatrix.
     * This is multiplied into the neural network .
     *
     *	@param dMatrix the dropout object to be set 
     */

    void setDropoutMatrix(ArrayList<ArrayList<Double>> dMatrix);

    /**
     * Propagates the input pattern through the network.
     *
     * @param inputPattern the input pattern for the network to process
     * @param dropout whether dropout is activated
     * @return the output of the network
     * @throws IllegalArgumentException if input pattern is null or does not match network input dimension
     */
    
    double[] feedForward(double[] inputPattern, boolean dropout);

}