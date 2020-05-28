package weka_2nd._02;



import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.WrapperSubsetEval;
import weka.core.Instances;

public class Agency_030SelectFeature {

	private ASEvaluation[] attrEval = null;
	private ASSearch[] searchMethod = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public Agency_030SelectFeature (){
		this.attrEval = new ASEvaluation[5];

		this.attrEval[0] = new WrapperSubsetEval();
		this.attrEval[1] = new CorrelationAttributeEval();
		this.attrEval[2] = new GainRatioAttributeEval();
		this.attrEval[3] = new InfoGainAttributeEval();
		this.attrEval[4] = new OneRAttributeEval();

		this.searchMethod = new ASSearch[2];
		this.searchMethod[0] = new BestFirst();
		this.searchMethod[1] = new Ranker();
	}


	 // �Ӽ�����
	public Agency_031FeatureSel_Result[] attrSelFeature(Instances data) throws Exception{
		Agency_031FeatureSel_Result[] result = null;
				
		// ��ǥ���� ����
		data.setClassIndex(data.numAttributes()-1);
		
		// �Ӽ����� �гΰ�ü ����
		AttributeSelection attrSelector = new AttributeSelection();

		result = new Agency_031FeatureSel_Result[this.attrEval.length]; // WrapperSubsetEval ����
		
		for(int x=0 ; x < this.attrEval.length ; x++){
			result[x] = new Agency_031FeatureSel_Result(data.numAttributes()); 
				
			// ���þ˰��� 
			attrSelector.setEvaluator(attrEval[x]);			
			
			// Ž�����
			if(x == 0)
				attrSelector.setSearch(searchMethod[0]); // BestFirst
			else
				attrSelector.setSearch(searchMethod[1]); // Ranker
					
			// �Ӽ����� �ǽ�
			attrSelector.SelectAttributes(data);
				
			// ������ �Ӽ� ��� ���� ��  ���
			result[x].setAttributeAlogrithms(attrEval[x]);
			result[x].setResult(attrSelector, data);
		}	
		return result;
	}
}
