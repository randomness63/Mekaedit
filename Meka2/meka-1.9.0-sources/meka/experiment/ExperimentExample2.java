/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * ExperimentExample.java
 * Copyright (C) 2015 University of Waikato, Hamilton, NZ
 */

package meka.experiment;

import mulanedit.BPMLLwD;
import meka.classifiers.multilabel.MULANedit;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.classifiers.multilabel.meta.BaggingML;
import meka.core.OptionUtils;
import meka.events.LogEvent;
import meka.events.LogListener;
import meka.experiment.datasetproviders.DatasetProvider;
import meka.experiment.datasetproviders.LocalDatasetProvider;
import meka.experiment.datasetproviders.MultiDatasetProvider;
import meka.experiment.evaluationstatistics.KeyValuePairs;
import meka.experiment.evaluators.CrossValidation;
import meka.experiment.evaluators.RepeatedRuns;
import meka.experiment.events.*;
import meka.experiment.statisticsexporters.*;
import moa.options.FlagOption;
import mulan.classifier.neural.BPMLL;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Utils;
import weka.core.pmml.jaxbbindings.ModelExplanation;

import java.io.File;
import java.io.ObjectInputStream.GetField;

import javax.xml.transform.Templates;

import junit.framework.Test;

/**
 * Just for testing the experiment framework.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class ExperimentExample2 {
	public static void main(String[] args) throws Exception {
		/*
		String tmpDir = System.getProperty("java.io.tmpdir");
		System.out.println("Using tmp dir: " + tmpDir);
		*/
		
		Experiment exp = new DefaultExperiment();
		String [] args2 = null;
		// classifiers

		MultiLabelClassifier model = new MULANedit();
		int temp2 = Utils.getOptionPos("E", args);
		int temp3 = Utils.getOptionPos("f", args);
		int temp4 = Utils.getOptionPos("q", args);
		if (Utils.getOptionPos("E", args)>= 0){
			model = new BaggingML();
			model.setOptions(args);
		}else {
			if (args != null){
				args2 = args;
			}else{
				String[] temp = {"-S","BPMLLwD2"};
				args2 = temp;
			}
			model.setOptions(args2);
		}
		
		exp.setClassifiers(new MultiLabelClassifier[]{
				model
		});
		// datasets
		LocalDatasetProvider dp1 = new LocalDatasetProvider();
		int temp = Utils.getOptionPos("t", args);
		dp1.setDatasets(new File[]{
				new File(args[temp+1]),
		});
		MultiDatasetProvider mdp = new MultiDatasetProvider();
		mdp.setProviders(new DatasetProvider[]{dp1});
		exp.setDatasetProvider(mdp);
		// output of metrics
		KeyValuePairs sh = new KeyValuePairs();
		sh.setFile(new File("mekaexp.txt"));
		exp.setStatisticsHandler(sh);
		// evaluation
		RepeatedRuns eval = new RepeatedRuns();
		if (Utils.getOptionPos("q", args)>= 0){
			eval.setUpperRuns(Integer.valueOf(args[temp4+1]));
		}else{
			eval.setUpperRuns(10);
		}
		CrossValidation cvtest = new CrossValidation();
		if (Utils.getOptionPos("f", args)>= 0){
			cvtest.setNumFolds(Integer.valueOf(args[temp3+1]));
		}else{
			cvtest.setNumFolds(10);
		}
		eval.setEvaluator(cvtest);
		exp.setEvaluator(eval);
		
		// stage
		exp.addExecutionStageListener(new ExecutionStageListener() {
			@Override
			public void experimentStage(ExecutionStageEvent e) {
				System.err.println("[STAGE] " + e.getStage());
			}
		});
		// iterations
		exp.addIterationNotificationListener(new IterationNotificationListener() {
			@Override
			public void nextIteration(IterationNotificationEvent e) {
				System.err.println("[ITERATION] " + Utils.toCommandLine(e.getClassifier()) + " --> " + e.getDataset().relationName());
			}
		});
		// statistics
		exp.addStatisticsNotificationListener(new StatisticsNotificationListener() {
			@Override
			public void statisticsAvailable(StatisticsNotificationEvent e) {
				System.err.println("[STATISTICS] #" + e.getStatistics().size());
			}
		});
		// log events
		exp.addLogListener(new LogListener() {
			@Override
			public void logMessage(LogEvent e) {
				System.err.println("[LOG] " + e.getSource().getClass().getName() + ": " + e.getMessage());
			}
		});
		// output options
		System.out.println("Setup:\n" + OptionUtils.toCommandLine(exp) + "\n");
		// execute
		String msg = exp.initialize();
		System.out.println("initialize: " + msg);
		if (msg != null)
			return;
		msg = exp.run();
		System.out.println("run: " + msg);
		msg = exp.finish();
		System.out.println("finish: " + msg);
		// export them
		TabSeparated tabsepAgg = new TabSeparated();
		int i = 0;
		boolean isExisting = true;
		while (isExisting){
			String num = Integer.toString(i);
			File checkFile = new File("mekaexp-agg"+ num +".tsv");
			if (checkFile.exists()){
				i++;
			}else{
				isExisting = false;
			}
		}
		String num = Integer.toString(i);
		tabsepAgg.setFile(new File("mekaexp-agg"+ num +".tsv"));
		SimpleAggregate2 aggregate = new SimpleAggregate2();
		aggregate.setSuffixMean("");
		aggregate.setSuffixStdDev(" (stdev)");
		aggregate.setSkipCount(true);
		aggregate.setSkipMean(false);
		aggregate.setSkipStdDev(false);
		aggregate.setExporter(tabsepAgg);
		TabSeparated tabsepFull = new TabSeparated();
		tabsepFull.setFile(new File("mekaexp-full" + num + ".tsv"));
		MultiExporter multiexp = new MultiExporter();
		multiexp.setExporters(new EvaluationStatisticsExporter[]{aggregate, tabsepFull});
		multiexp.addLogListener(new LogListener() {
			@Override
			public void logMessage(LogEvent e) {
				System.err.println("[EXPORT] " + e.getSource().getClass().getName() + ": " + e.getMessage());
			}
		});
		System.out.println(OptionUtils.toCommandLine(multiexp));
		msg = multiexp.export(exp.getStatistics());
		System.out.println("export: " + msg);
	}
}
