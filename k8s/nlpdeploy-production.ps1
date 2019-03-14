
Param(
    [parameter(Mandatory=$false)][string]$registry="botbotregistry.azurecr.io",
    [parameter(Mandatory=$false)][string]$dockerUser="botbotregistry",
    [parameter(Mandatory=$false)][string]$dockerPassword="EhLteirZVvzHcZ9+u3w1FYn+Im4gfrRy",
    [parameter(Mandatory=$true)][string]$imageTag,
    [parameter(Mandatory=$false)][string]$dockerOrg="botbot"
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
az aks get-credentials --resource-group botbot-k8s-group --name botbot-k8s
    
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