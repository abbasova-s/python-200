1. 
ReadAlertSettings: Allows reading one or more parameters stored under the ctd/submission-alert/ path in AWS Systems Manager Parameter Store.

BlockAdminSettings: Explicitly denies all Systems Manager actions on resources under the ctd/submission-alert/admin/ path.

PublishSubmissionMetric: Allows publishing metrics to Amazon CloudWatch. The resource is * because this API does not support restricting this action to a specific resource ARN.

WriteOwnLogs: Allows creating log streams and writing log events only in the submission-alert Lambda log group.

2. 
It uses * because CloudWatch's PutMetricData action does not support resource-level permissions, so a specific resource ARN cannot be used. The statement still follows least privilege because it allows only one specific action instead of giving access to all CloudWatch operations.

3. 
Changing the policy to allow ssm:* on all resources would give the alert much more access than it needs. It could modify or delete parameters, including admin settings, instead of only reading the alert settings. This violates the principle of least privilege and removes the protection provided by the explicit Deny statement.