package mulanedit;
import java.util.Random;

import javax.crypto.SealedObject;

import org.jgap.util.randomHotBits;

import java.util.ArrayList;
import java.util.Collections;

//This file contains the methods for creating, printing and modifying the dropout matrix
//some methods are deprecated

public class Dropout {
	ArrayList<ArrayList<Double>> dropoutMatrix;
	int [] netTopology;
	double ratio;
	
	public Dropout(int[] netTopology, double ratio){
		this.netTopology = netTopology;
		this.ratio = ratio;
		
		createNewMatrix();
		
	}
	
	//returns dropout matrix
	public ArrayList<ArrayList<Double>> getDropoutMatrix(){
		return dropoutMatrix;
	}
	
	
	//creates 2d matrix for dropout
	public void createNewMatrix (){
		dropoutMatrix = new ArrayList<ArrayList<Double>>();
		
		//reads netTopology for size of each layer and outputs dropout lists of the same size
		//adds these layers to a list
    	for (int layer=0;layer < netTopology.length - 1;layer++){
    		ArrayList<Double> layerDropout = createDropout(ratio, netTopology[layer]);
    		dropoutMatrix.add(layerDropout);
    	}
    	
    	//for last layer change all to ones
    	ArrayList<Double> layerDropout = createDropout(1,netTopology[netTopology.length-1]);
    		dropoutMatrix.add(layerDropout);
	}
	
	public void printDropout(){
    	for (int i=0; i<dropoutMatrix.size();i++){
    		System.out.println(dropoutMatrix.get(i));
    		System.out.println("-");
    	}
    	
	}	
	
	//creates a double list with 1/dropout chance for some fraction of members
	//and 0 for the rest
	//shuffles the collection
	//might not be efficient?
	public ArrayList<Double> createDropout(double dropoutProportion, int matrixLength){
		int numTrue = (int)(dropoutProportion * matrixLength);
		ArrayList<Double> dropoutMatrix = new ArrayList<Double>();
		for (int i = 0; i < matrixLength; i++){
			if (i < numTrue){
				dropoutMatrix.add(1/dropoutProportion);  
			}else{
				dropoutMatrix.add(0.0);
			}
		}
		
		Collections.shuffle(dropoutMatrix);
		
		return dropoutMatrix;
	}
	
	//alternative function for creating dropout, does not ensure that proportion active is constant
	public ArrayList<Double> createRandomDropout(double dropoutProportion, int matrixLength){
		int numTrue = (int)(dropoutProportion * matrixLength);
		ArrayList<Double> dropoutMatrix = new ArrayList<Double>();
		Random random = createRandom();
		for (int i = 0; i < matrixLength; i++){
			double dropout = 0.0;
			
			if (bernoulli(dropoutProportion, random)){
				dropout = 1/dropoutProportion;
			}
			
			dropoutMatrix.add(dropout);
		}
		
		return dropoutMatrix;
	}
	
    private Random createRandom(){
        return new Random();
    }

    private double uniform (Random random){
        return random.nextDouble();
    }

    private boolean bernoulli (double p, Random random){
        return uniform(random) < p;
    }
   
}