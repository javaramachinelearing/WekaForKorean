package weka_2nd._05.bird;

import java.util.Random;
import org.apache.commons.math3.stat.descriptive.AggregateSummaryStatistics;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Remove;
import weka_2nd.MoreWekaCommon;

public class Image_030PerceptronClassifier {

	private FilteredClassifier model = null;
	private Classifier innerModel = null;

	public Image_030PerceptronClassifier(){
		this.model = new FilteredClassifier();
		this.innerModel = new J48();
	}

	public Image_030PerceptronClassifier(Classifier cls){
		if( cls instanceof J48)
			this.innerModel = new J48();
		else if( cls instanceof MultilayerPerceptron)
			this.innerModel = new MultilayerPerceptron();
			
		this.model = new FilteredClassifier();
	}
	
	public void saveModel(Classifier model) throws Exception{
		weka.core.SerializationHelper.write(MoreWekaCommon.modelPath +"image\\image_bird_" +
				                            MoreWekaCommon.getModelName(model), model);
	}
	
	public Classifier buildModel(Instances data) throws Exception{
		
		this.crossValidataion(data);		
		
		// ������ ����
		this.saveModel(model);
		
		// ������ ��ȯ
		return model;
	}
	
	public double crossValidataion(Instances data) throws Exception{
		int numfolds = 10;
		int numfold = 0;
		  
		// 1) data loader 		
		Instances train = data.trainCV(numfolds, numfold, new Random(10));
		Instances test  = data.testCV (numfolds, numfold);
		
		// 2) class assigner
		train.setClassIndex(train.numAttributes()-1); 
		test.setClassIndex(test.numAttributes()-1);  
		
		// 3) set innerClassifer, filters
		model.setClassifier(innerModel);
		model = this.setFilteredClassifier(model,data);
		  
		// 4) cross validate setting  
		Evaluation eval=new Evaluation(train);	
		eval.crossValidateModel(model, train, numfolds, new Random(10));	  

		// 5) model run 
		model.buildClassifier(train);
		   
		// 6) evaluate
		eval.evaluateModel(model, test);
		
		// 7) print Result text 
		this.printResult(eval , test);
						
		return eval.pctCorrect();
	}
	
	public void printResult(Evaluation eval, Instances test) throws Exception{
		
		System.out.println(MoreWekaCommon.getModelName(this.model.getClassifier()) + " : "  +
				   "�з���� ������ �� �� : " + (int)eval.numInstances() + 
		           ", ���з� �Ǽ� : " + (int)eval.correct() + 
		           ", ���з��� : " + String.format("%.1f",eval.correct() / eval.numInstances() * 100) +" %" +
		           ", \n ���з�ǥ : \n" +  eval.toMatrixString() +
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

	// FilteredClassifier ���� ���� �� ���� �з��˰��� ����
	public FilteredClassifier setFilteredClassifier(FilteredClassifier filterCls, Instances data) throws Exception{
		MultiFilter mFilter = new MultiFilter();
		Filter[] filters = new Filter[2];
				
		filters[0] = new Image_021Filter(data).setImageFilters()
				                              .getImgFilter();

		Remove remove = new Remove();
		remove.setAttributeIndices("first");
		filters[1] = remove;
				
		mFilter.setFilters(filters);
		
		filterCls.setFilter(mFilter);		
		
		return filterCls;
	}

}
