package weka_2nd._03;

import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka_2nd.MoreWekaCommon;

public class VoC_010Dataset {
	
	private Instances voc_sentence= null;
	private Instances voc_word= null;
	private Instances s2vec_voc_sentence= null;
	private Instances s2vec_voc_word= null;
	private Classifier model = null;
	
	public static void  main (String args[]) throws Exception{
		VoC_010Dataset obj = new VoC_010Dataset();
		obj.openArffonWeb();
		obj.preprocess();
		obj.build();
		obj.classify();
	}

	// 1) data loader
	public void openArffonWeb() throws Exception{
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(MoreWekaCommon.arffPath+"\\textMining\\voc_sentence.arff"));
		this.voc_sentence = loader.getDataSet();
		System.out.println("전체 데이터 건 수 : " + voc_sentence.size());

		loader.setFile(new File(MoreWekaCommon.arffPath+"\\textMining\\voc_word.arff"));
		this.voc_word = loader.getDataSet();
		System.out.println("전체 데이터 건 수 : " + voc_word.size());
	}
	
	// 2) 전처리 (각종 필터링)
	public void preprocess() throws Exception{
		VoC_020Preprocess preprocess 
		         = new VoC_020Preprocess(this.voc_sentence,  this.voc_word);
		
		Instances[] s2vec_data = preprocess.filter();	
		
		s2vec_data[0].setRelationName("문장 s2vec");
		this.setS2vec_voc_sentence(s2vec_data[0]);
		
		s2vec_data[1].setRelationName("단어 s2vec");
		this.setS2vec_voc_word(s2vec_data[1]);
	}
	

	// 3) 모델 학습/평가/저장
	public void build() throws Exception{
		VoC_030FilteredClassifier textMining = new VoC_030FilteredClassifier(new J48());
		this.model = textMining.buildModel(this.voc_sentence);

		textMining = new VoC_030FilteredClassifier(new NaiveBayesMultinomial());
		this.model = textMining.buildModel(this.voc_sentence);
	}
	
	// 4) 모델 분류
	public void classify() throws Exception{
		VoC_040Classify classify 
        = new VoC_040Classify(this.model);

		System.out.println("**********************");
		System.out.println("문장 학습 + 단어 분류");
		System.out.println("**********************");
		classify.useModel(this.voc_word);
		
	}

	public Instances getS2vec_voc_sentence() {
		return s2vec_voc_sentence;
	}

	public void setS2vec_voc_sentence(Instances s2vec_voc_sentence) {
		this.s2vec_voc_sentence = s2vec_voc_sentence;
	}

	public Instances getS2vec_voc_word() {
		return s2vec_voc_word;
	}

	public void setS2vec_voc_word(Instances s2vec_voc_word) {
		this.s2vec_voc_word = s2vec_voc_word;
	}
	
}
