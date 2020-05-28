package weka_2nd._01;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RenameAttribute;
import weka.filters.unsupervised.attribute.RenameNominalValues;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.Standardize;
import weka.filters.unsupervised.attribute.StringToNominal;

public class Titanic_021Filter {	

	private Instances data = null;

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	public Titanic_021Filter (Instances data){
		this.data = data;
	}
	
	public Instances action() throws Exception{
		this.renameAttributes(); // �Ӽ����� �ѱ۷� ġȯ
		this.numericToNominal("1"); // ������ �����͸� ��������� ��ȯ
		this.renameValues();     // ������ ���� �Ϻθ� �ѱ۷� ġȯ 
		this.remove("10,13,14"); // ������ 33% �̻� �Ӽ� ���� (��,��������ȣ �Ӽ��� ����) 
		this.reOrder("1,3-last,2");          // �Ӽ���迭 (2��° �������� �Ӽ��� �� �ڷ� ��ġ)
		this.stringToNominal("2,7,10");  // String�� �����͸� ��������� ��ȯ 
		return this.data;
	}	
	
	public Instances normalize () throws Exception{
		Normalize normalize = new Normalize();
		runFilter(normalize);
		return this.data;
	}

	public Instances standardize () throws Exception{
		Standardize standardize = new Standardize();
		runFilter(standardize);
		return this.data;
	}
	
	/******************
	 * �Ӽ����� �ѱ۷� ��ȯ
	 ******************/
	public void renameAttributes () throws Exception{
		this.setRenameAttributeFiler("1", "���ǵ��");
		this.setRenameAttributeFiler("2", "��������");
		this.setRenameAttributeFiler("3", "����");
		this.setRenameAttributeFiler("4", "����");
		this.setRenameAttributeFiler("5", "����");
		this.setRenameAttributeFiler("6", "����_�����_������");
		this.setRenameAttributeFiler("7", "����_�θ�_�ڳ��");
		this.setRenameAttributeFiler("8", "ticket��ȣ");
		this.setRenameAttributeFiler("9", "���");
		this.setRenameAttributeFiler("10", "���ǹ�ȣ");
		this.setRenameAttributeFiler("11", "ž����ġ");
		this.setRenameAttributeFiler("12", "��������ȣ ");
		this.setRenameAttributeFiler("13", "����");
		this.setRenameAttributeFiler("14", "������");
	}
	
	public void setRenameAttributeFiler(String index, String Replace) throws Exception{
		RenameAttribute renameAttr = new RenameAttribute();
		renameAttr.setAttributeIndices(index);
		renameAttr.setReplace(Replace);
		this.runFilter(renameAttr);
	}
	
	public void renameValues () throws Exception{
		this.setRenameValueFilter("1","1:1���,2:2���,3:3���");// NumericToNominal ������ ���͸�
		this.setRenameValueFilter("2","1:����,0:���");
		this.setRenameValueFilter("4","female:����,male:����");
	}
	
	public void setRenameValueFilter(String index, String Replacement) throws Exception{
		RenameNominalValues renameValue = new RenameNominalValues();		
		renameValue.setSelectedAttributes(index);
		renameValue.setValueReplacements(Replacement);
		this.runFilter(renameValue);
	}

	public void numericToNominal(String index) throws Exception{
		NumericToNominal numToNom = new NumericToNominal();	
		numToNom.setAttributeIndices(index);
		this.runFilter(numToNom);	
	}

	public void stringToNominal(String index) throws Exception{
		StringToNominal strToNom = new StringToNominal();		
		strToNom.setAttributeRange(index);
		this.runFilter(strToNom);
	}
	
	public void reOrder(String index) throws Exception{
		Reorder strToNom = new Reorder();		
		strToNom.setAttributeIndices(index);
		this.runFilter(strToNom);
	}

	public Instances remove(String index) throws Exception{
		Remove remove = new Remove();		
		remove.setAttributeIndices(index);
		this.runFilter(remove);
		return this.data;
	}
	
	public Instances runFilter(Filter filter) throws Exception {
		filter.setInputFormat(data);	
		data = Filter.useFilter(data, filter);	
		return this.data;
	}
	
	public Instances getData() {
		return this.data;
	}
}
