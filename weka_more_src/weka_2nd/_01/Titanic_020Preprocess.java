package weka_2nd._01;

import java.awt.Dimension;

import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.JTable;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Instance;
import weka.core.Instances;

public class Titanic_020Preprocess {

	private Instances data = null;
	private Instances titanic_co_reorder = null;
	private Instances titanic_co_reorder_norm = null;
	private Instances titanic_co_reorder_stan = null;
	private Instances titanic_co_reorder_min = null;
	private Instances titanic_co_reorder_attr = null;
	
	private Titanic_021Filter filter = null;

	public Titanic_020Preprocess (Instances data){
		this.data = data;
	}

	public Instances getReorder(){return this.titanic_co_reorder;}
	public Instances getNomalize(){return this.titanic_co_reorder_norm;}
	public Instances getStandardize(){return this.titanic_co_reorder_stan;}
	public Instances getMin(){return this.titanic_co_reorder_min;}
	public Instances getSelectedAttr(){return this.titanic_co_reorder_attr;}
	
	public void filter() throws Exception{
		this.showRules("original data set",data);
		this.filter = new Titanic_021Filter(this.data);
		this.titanic_co_reorder = filter.action();
		this.showRules("�Ӽ� ��迭", titanic_co_reorder);

		// ����ȭ
		this.titanic_co_reorder_norm = new Titanic_021Filter(titanic_co_reorder).normalize();
		this.showRules("�Ӽ� ��迭 + ����ȭ", titanic_co_reorder_norm);

		// ǥ��ȭ
		this.titanic_co_reorder_stan = new Titanic_021Filter(titanic_co_reorder).standardize();
		this.showRules("�Ӽ� ��迭 + ǥ��ȭ", titanic_co_reorder_stan);

		// �Ӽ� ���� ����
		this.titanic_co_reorder_min = new Titanic_021Filter(titanic_co_reorder).remove("2,7");
		this.showRules("�Ӽ� ��迭 + ���ǻ���", titanic_co_reorder_min);
		
		// �Ӽ�����
		this.titanic_co_reorder_attr = this.attrSelClassifier(titanic_co_reorder);
		this.showRules("�Ӽ� ��迭 + �Ӽ�����", titanic_co_reorder_attr);
	}	

	// swing ���̺� ���� ����
	 public void showRules(String title, Instances filteredData) throws Exception{

		 // �Ӽ��� ���� (header)
		 String header[] = new String[filteredData.numAttributes()];

		 for (int i = 0; i < filteredData.numAttributes() ; i++) 
			header[i] = filteredData.attribute(i).name();
		 
		 // ������ �� ���� (contents)
		 String contents[][] = new String[filteredData.numInstances()][filteredData.numAttributes()];

		 for(int x=0 ; x < filteredData.numInstances(); x++){
			 Instance row = filteredData.get(x);
			 for (int y = 0; y < row.numAttributes(); y++) {
				try{
					contents[x][y] = row.stringValue(y);
				}catch (IllegalArgumentException iae){
					contents[x][y] = row.value(y)+"";
				}	
			}
		}
		this.makeTable(title, header, contents);
	 }
	 
	 // swing ���̺�� ��� ���
	 public void makeTable(String title, String[] header, String[][] contents){
		 Dimension dim = new Dimension(1024,768);
		 JFrame frame = new JFrame(title);
		 frame.setLocation(100, 200);
		 frame.setPreferredSize(dim);
		 
		 JTable table = new JTable(contents, header);
		 JScrollPane scrollpane = new JScrollPane(table);
		
		 frame.add(scrollpane);
		 frame.pack();
		 frame.setVisible(true);
	 }
	 

	 // �Ӽ�����
	public Instances attrSelClassifier(Instances data) throws Exception{
				
		// ��ǥ���� ����
		data.setClassIndex(data.numAttributes()-1);
		
		// �Ӽ����� �гΰ�ü ����
		AttributeSelection attrSelector = new AttributeSelection();
		
		// ������ �Ӽ����� ����
		CorrelationAttributeEval attrEval = new CorrelationAttributeEval();
		Ranker search = new Ranker();
		search.setThreshold(0.2); // �Ӱ谪 ����
		attrSelector.setEvaluator(attrEval);
		attrSelector.setSearch(search);
				
		// �Ӽ����� �ǽ�
		attrSelector.SelectAttributes(data);
		
		// ������ �Ӽ��� ��迭
		Instances attrSelect = attrSelector.reduceDimensionality(data);
		return attrSelect;
	}

}
