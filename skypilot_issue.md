## SkyPilot managed jobs fail with "Invalid zone" error on RunPod

**Version**: SkyPilot 0.10.0+

**Issue**: When launching managed jobs that auto-select RunPod, the job controller fails with:
```
ValueError: Invalid zone 'CA-MTL-1,CA-MTL-2,CA-MTL-3' for region 'CA'
```

**Reproduction**:
```bash
# job.yaml
name: test-job
resources:
  accelerators: A100-80GB:1
  use_spot: false
run: echo "test"

# Launch command
sky jobs launch job.yaml
```

**Expected**: Job should launch successfully on RunPod or fallback to another provider

**Actual**: Job controller crashes with ValueError before provisioning

**Workaround**: Explicitly specify `cloud: <provider>` to avoid RunPod selection

This appears to be a zone validation issue in the managed jobs controller when RunPod is selected with Canadian regions.