"""Shared GPU slot accounting while the prior queue drains its already launched jobs."""

def prior_dispatch_finished(queue,now):
    if queue.get('failed'):return False
    if queue.get('status')=='COMPLETE':return True
    return (queue.get('status')=='RUNNING' and not queue.get('pending')
            and now-queue.get('updated',0)<=90)

def can_launch(*,total,free,host_free,own_jobs,prior_jobs,cost,prior_cost):
    return (own_jobs+prior_jobs<2 and host_free>=12.
            and free-cost>=max(4.,.15*total)
            and (own_jobs+1)*cost+prior_jobs*prior_cost<=.8*total)
