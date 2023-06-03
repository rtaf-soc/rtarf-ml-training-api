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
                        withCredentials([usernamePassword(credentialsId: 'AWS_CREDENTIAL', passwordVariable: 'password', usernameVariable: 'username')])     
                        {
                            sh "mc alias set myminio https://minio-ml-hl.minio-ml.svc.cluster.local:9000 ${username} ${password}"
                            sh "mkdir rawdata"
                            sh "mc cp --recursive  myminio/data-from-siem/14/ ./rawdata/"
                        }                                                                            
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
                        sh "python --version"                           
                        sh "python train.py"                           
                    }
                }
            }
        }         
    }
}