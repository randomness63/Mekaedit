package mulanedit;
import static org.junit.Assert.*;

import org.junit.Test;

public class Experiment {

	@Test
	public void test() throws Exception {
		String[] arg = "-arff emotions.arff -xml emotions.xml -unlabeled emotions.arff".split(" ");
		MulanExp.main(arg);
		
	}

}
