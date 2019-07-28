import java.io.*;
import java.util.*;
import java.util.Properties;

import edu.stanford.nlp.io.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.*;
import edu.stanford.nlp.util.*;
//import edu.stanford.nlp.pipeline.StanfordCoreNLP;
//import edu.stanford.nlp.pipeline.Annotation;
//import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
//import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.trees.CollinsHeadFinder;
//import edu.stanford.nlp.trees.international.pennchinese.ChineseHeadFinder;


public class Test2
{
	public static void main(String args[])
	{
		
		 Trees.PennTreeReader reader = new Trees.PennTreeReader(new StringReader("((S (NP (DT the) (JJ quick) (JJ (AA (BB (CC brown)))) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))) (. .)))"));
    Tree<String> tree = reader.next();
    System.out.println("tree " + tree);
    CollinsHeadFinder headFinder = new CollinsHeadFinder();
   
        Tree<String> head = headFinder.determineHead(tree);
        System.out.println("head " + head);
      
    
		
	}
}