from easydict import EasyDict as edict

config = edict()

config.data_dir = "../r6.2"
config.SPARK_MASTER = 'spark://node5:7077'
config.regular_hours = [7, 20] # regular working hours in 24 hours format.
config.org_domain = "dtaa.com" # domain used by an organization.
config.user_pcs = "./user_pcs.txt"
config.test_start_month = [2011,03]
config.cache_dir = "../cache"
config.job_sites = [
    "linkedin.com",
    "glassdoor.com",
    "indeed.com",
    "careerbuilder.com",
    "simplyhired.com",
    "usajobs.gov",
    "linkup.com",
    "snagajob.com",
    "roberthalf.com",
    "dice.com",
    "idealist.com",
    "monster.com",
    "us.jobs",
    "collegerecruiter.com",
    "coolworks.com",
    "efinancialcareers.com",
    "energyjobline.com",
    "engineering.jobs",
    "ecojobs.com",
    "flexjobs.com",
    "healthcarejobsite.com",
    "internships.com",
    "mediabistro.com",
    "onewire.com",
    "jobs.prsa.org",
    "salesgravy.com",
    "salesjobs.com",
    "snagajob.com",
    "talentzoo.com",
    "youtern.com"]
config.cloudstorage_sites = [
    "dropbox.com",
    "wikileaks.org",
    "drive.google.com",
    "onedrive.live.com",
    "box.com",
    "carbonite.com",
    "mega.nz",
    "icloud.com",
    "spideroak.com",
    "idrive.com",
    "pcloud.com"]

