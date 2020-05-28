package weka_2nd._01;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.stat.inference.TTest;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka_2nd.MoreWekaCommon;

public class Titanic_030Experiment {

	private Classifier[] model = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public Titanic_030Experiment (){
		this.model = new Classifier[4];

		this.model[0] = new J48();
		this.model[1] = new SMO();
		this.model[2] = new NaiveBayes();
		this.model[3] = new OneR();
		
	}
	
	public void saveModel(Classifier model) throws Exception{
		weka.core.SerializationHelper.write(MoreWekaCommon.modelPath +"titanic\\titanic_" +
				                            MoreWekaCommon.getModelName(model), model);
	}
	
	public Classifier experimentModel(Instances data) throws Exception{
		int n = 30;
		double sum[][] = new double[this.model.length][n] ;
		
		// �𵨺� ���з� ���� (�𵨺� 30�� �̻�)
		for(int i=0 ; i < this.model.length ; i++){
			System.out.println(MoreWekaCommon.getModelName(this.model[i]));
			for(int seed=0 ; seed < n ; seed++)
				sum[i][seed] = this.crossValidataion(data, this.model[i], seed+1);
		}
		
		// �𵨺� ���з��� ��� �� ���� ����
		Titanic_031ExpResult[] expResult =  
				new Titanic_031ExpResult[this.model.length];		
		for(int i=0 ; i < this.model.length ; i++){
			System.out.print(MoreWekaCommon.getModelName(this.model[i]) + " : ");
			expResult[i] = new Titanic_031ExpResult();
			expResult[i].setModel(this.model[i]);
			expResult[i].setPctCorrect(this.aggregateValue(sum[i]));
			expResult[i].setPctCorrects(sum[i]);
		}	
		
		// ��� ���з��� ���� �˰��� ��� ���� (���з����� ���� ���� �ڷ� ��ġ)
		Arrays.sort(expResult);		

		// �𵨺� ������ ���� (���м� t-test, ���Ǽ��� 0.05)
		for(int i=0 ; i < expResult.length-1 ; i++){
			System.out.print(
					MoreWekaCommon.getModelName(expResult[expResult.length-1].getModel()) +
					" vs. " + 
					MoreWekaCommon.getModelName(expResult[i].getModel()) + 
					" : "
					);
			// ���з��� ����� t-����
			this.t_test(expResult[expResult.length-1].getPctCorrects(), expResult[i].getPctCorrects());			
		}
		
		// ������ ����
		this.saveModel(expResult[expResult.length-1].getModel());
		
		// ������ ��ȯ
		return expResult[expResult.length-1].getModel();
	}
	
	public double crossValidataion(Instances data, Classifier model , int seed) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		  
		// 1) data loader 
		data.setClassIndex(data.numAttributes()-1); 
		
		Instances train = data.trainCV(numfolds, numfold, new Random(seed));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		  
		// 3) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		eval.crossValidateModel(model, train, numfolds, new Random(seed));	  

		// 4) model run 
		model.buildClassifier(train);
		   
		// 5) evaluate
		eval.evaluateModel(model, test);
		
		// 6) print Result text 
		this.printResult(eval);
				
		return eval.pctCorrect();
	
	}
	
	public void printResult(Evaluation eval) throws Exception{
		System.out.println("\t�з���� ������ �� �� : " + (int)eval.numInstances() + 
		           ", ���з� �Ǽ� : " + (int)eval.correct() + 
		           ", ���з��� : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" +
//		           ", \n ���з�ǥ : \n" +  eval.toMatrixString() +
		           ""
					); 	
	}
	
	/**
	 * common-math jar �ٿ�ε� ��ġ : http://apache.mirror.cdnetworks.com/commons/math/binaries/
	 * **/
	public double aggregateValue(double[] sum){
		AggregateSummaryStatistics aggregate = new AggregateSummaryStatistics();
		SummaryStatistics sumObj = aggregate.createContributingStatistics();
		for(int i = 0; i < sum.length; i++)  
			sumObj.addValue(sum[i]); 

		System.out.println("��� : " + String.format("%.1f",aggregate.getMean()) 
		             + " %, ���� : " + String.format("%.1f",aggregate.getStandardDeviation()));
		
		return aggregate.getMean();
	}
	
	public void t_test(double[] criterion, double[] sample){
		TTest t_test = new TTest();
		System.out.print( " �Ⱒ�� ���� ���� : "
				         + t_test.tTest(criterion, sample, 0.05)
				         + " ==> p-value : " 
						 +  t_test.tTest(criterion, sample)
                         + " , Experimenter ���� ǥ�� : " );
		
		double criterionMean = this.aggregateValue(criterion);
		double sampleMean = this.aggregateValue(sample);
		
		if (criterionMean > sampleMean)
			System.out.println((t_test.tTest(criterion, sample, 0.05)?"*":""));
		else
			System.out.println((t_test.tTest(criterion, sample, 0.05)?"V":""));
	}

}
