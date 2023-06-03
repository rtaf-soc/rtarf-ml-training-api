pipeline 
{
    agent 
    {
        node 
        {
            label "jenkins-python"
        }       
    }

    stages
    {
        
        stage('PrePare ENV') 
        { 
            steps 
            {
                script 
                {
                    container("python") 
                    {
                        sh "python --version"                           
                        sh "pip install -r requirements.txt"                                                                              
                    }
                }
            }
        }         
        stage('Test Python') 
        { 
            steps 
            {
                script 
                {
                    container("python") 
                    {
                        sh "export MLFLOW_TRACKING_URI=http://mlflow.rtarf-ml.its-software-services.com/"                           
                        sh "python train.py"                           
                    }
                }
            }
        }         
    }
}