Param(
    [parameter(Mandatory=$false)][string]$registry="botbotregistrystagingsea.azurecr.io",
    [parameter(Mandatory=$false)][string]$dockerUser="botbotregistrystagingsea",
    [parameter(Mandatory=$false)][string]$dockerPassword="GTsb0Apf46WXE3Y=aqi=pi7JNRZ8rKc0",
    [parameter(Mandatory=$true)][string]$imageTag,
    [parameter(Mandatory=$false)][string]$configFile="conf_staging.yml",
    [parameter(Mandatory=$false)][string]$dockerOrg="botbot-staging"
)
function ExecKube($cmd) {    
    if($deployCI) {
        $kubeconfig = $kubeconfigPath + 'config';
        $exp = $execPath + 'kubectl ' + $cmd + ' --kubeconfig=' + $kubeconfig
        Invoke-Expression $exp
    }
    else{
        $exp = $execPath + 'kubectl ' + $cmd
        Invoke-Expression $exp
    }
}
# Initialization
az aks get-credentials --resource-group=botbot-k8s-group-staging-sea --name=botbot-k8s-staging-sea
Write-Host "Building Docker images tagged with '$imageTag'" -ForegroundColor Yellow
    $env:TAG=$imageTag
    
    docker-compose -p .. -f ../docker-compose.yml -f ../docker-compose.staging.yml build    

    Write-Host "Pushing images to $registry/$dockerOrg..." -ForegroundColor Yellow
    $services = ("botbot.nlp")

    foreach ($service in $services) {
        $imageFqdn = "$registry/$dockerOrg/${service}"
        docker tag botbot/${service}:$imageTag ${imageFqdn}:$imageTag
        docker push ${imageFqdn}:$imageTag            
    }


    
# Removing previous services & deployments
Write-Host "Removing existing services & deployments.." -ForegroundColor Yellow
ExecKube -cmd 'delete deployments botbotnlp --namespace botbot'
ExecKube -cmd 'delete services botbotnlp --namespace botbot'

Write-Host 'Deploying code deployments (Web APIs, Web apps, ...)' -ForegroundColor Yellow
ExecKube -cmd 'create -f nlpService.yaml'
Write-Host "Deploying configuration from $configFile" -ForegroundColor Yellow
ExecKube -cmd "create -f $configFile"

Write-Host "Creating deployments..." -ForegroundColor Yellow
ExecKube -cmd 'create -f nlpDeploy.yaml'  
# update deployments with the correct image (with tag and/or registry)
$registryPath = ""
if (-not [string]::IsNullOrEmpty($registry)) {
    $registryPath = "$registry/"
}

Write-Host "Update Image containers to use prefix '$registry$dockerOrg' and tag '$imageTag'" -ForegroundColor Yellow
ExecKube -cmd 'set image deployments/botbotnlp botbotnlp=${registryPath}${dockerOrg}/botbot.nlp:$imageTag --namespace botbot'


Write-Host "dashboard staging is exposed at http://$externalDns" -ForegroundColor Yellow