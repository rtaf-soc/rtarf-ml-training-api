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
        stage('PrePare DataSet') 
        { 
            steps 
            {
                script 
                {
                    container("minio-mc") 
                    {
                        sh "mc -v"                                                                            
                    }
                }
            }
        }         
        stage('Run Model') 
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