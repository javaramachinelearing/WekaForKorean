package weka_2nd._03;

import java.io.File;

import weka.core.Instances;
import weka.core.stopwords.WordsFromFile;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka_2nd.MoreWekaCommon;
import weka_2nd._01.Titanic_021Filter;

public class VoC_021Filter extends Titanic_021Filter {	
	private StringToWordVector stw = null;

	public VoC_021Filter(Instances data) {
		super(data);
	}
	
	public Instances action() throws Exception{
		return super.getData();
	}	
		
	
	public Instances getData() {
		return super.getData();
	}
	
	/***
	 * https://riptutorial.com/ko/weka/example/25415/liblinear%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%9C-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%B6%84%EB%A5%98
	 * 
	 */
	
	public VoC_021Filter setString2WorkVector() throws Exception {
		this.stw = new StringToWordVector();
		
		// ���¼�
		WordTokenizer token = new WordTokenizer();
		token.setDelimiters(" .,;:'()?!^~�ʽ��ϴ�����");
//		token.setDelimiters(" .,;:'()?!�ʽ��ϴ��ְ����");
		this.stw.setTokenizer(token);
		WordsFromFile stop = new WordsFromFile();
		stop.setStopwords(new File(MoreWekaCommon.arffPath+"\\textMining\\stop.txt"));
		this.stw.setStopwordsHandler(stop);

		//��ҹ��� ����
		this.stw.setLowerCaseTokens(true);
		
		// �ܾ ��ġȭ
		this.stw.setOutputWordCounts(true);
		this.stw.setTFTransform(true);
		this.stw.setIDFTransform(true);
		
		super.runFilter(this.stw);
		
		return this;
	}

	public StringToWordVector getStw() {
		return this.stw;
	}

	public void setStw(StringToWordVector stw) {
		this.stw = stw;
	}
}
