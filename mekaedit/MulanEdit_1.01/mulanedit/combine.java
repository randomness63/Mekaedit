package mulanedit;

import java.io.File;
import java.awt.print.Printable;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import javax.security.auth.x500.X500Principal;

import org.jgap.gp.function.If;
import org.junit.rules.TemporaryFolder;

public class combine {
	
	void main(){
	
		String filename = "nus-wide-full-cVLADplus-train.arff";
		String filename2 = ".arff";
	
		File file1 = new File(filename);
		File file2 = new File(filename2);
		StringBuilder rString = new StringBuilder();
		String fname = "nuswide.arff";
	
		if(file1.exists()){
			try (BufferedReader br = new BufferedReader(new FileReader(file1))) {
			    for(String line; (line = br.readLine()) != null; ) {
			        rString.append(line);
			    }
			}catch (Exception e) {
				// TODO: handle exception
			}
		}
		
		if(file2.exists()){
			try (BufferedReader br = new BufferedReader(new FileReader(file2))) {
			    for(String line; (line = br.readLine()) != null; ) {
			        rString.append(line);
			    }
			}catch (Exception e) {
				// TODO: handle exception
			}
		}
				
		String x = rString.toString();
		
		try{
			PrintWriter outer = new PrintWriter(new BufferedWriter(new FileWriter(fname)));
			outer.write(x);
			outer.close();
		}catch(Exception e){
			
		}

		
	}
	
	
	
	
}
