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
        
        stage('Test Python') 
        { 
            steps 
            {
                script 
                {
                    container("python") 
                    {
                        sh "python --version"                           
                    }
                }
            }
        }         
    }
}