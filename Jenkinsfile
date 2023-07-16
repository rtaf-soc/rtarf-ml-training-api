MLFLOW_MINIO_FOLDER=env.MLFLOW_MINIO_FOLDER

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
                            // sh "mc cp myminio/data-from-siem/14/ls.s3.b504175e-ca7b-48c5-8551-4ffdc251ed75.2023-05-14T23.07.part1358.txt rawdata/1.txt"
                            sh "mc cp --recursive  myminio/data-from-siem/${MLFLOW_MINIO_FOLDER}/ ./rawdata/"
                            sh "ls -alrt ./rawdata/"
                        }                                                                            
                    }
                }
            }
        }         
        stage('Run Model DST') 
        { 
            steps 
            {
                script 
                {
                    container("python") 
                    {
                        sh "python train-dst.py"
                    }
                }
            }
        }

        stage('Run Model TIME') 
        { 
            steps 
            {
                script 
                {
                    container("python") 
                    {
                        sh "python train-time.py"
                    }
                }
            }
        }         
    }
}