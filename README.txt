Microduck Stage12 local recovery v6 evidence transport.
Separate transfer branch; never merge into double-balance.
Read-only v5 raw-evidence recovery; all native and training budgets are zero.
Read return-manifest.json, every attachment manifest, and all their parts at one full commit SHA.
Verify encoded SHA/size, strict base64 round trip, decoded SHA/size,
reconstructed ZIP SHA/size and RETURN-MANIFEST.json before review.
Pushing is manual. A transfer commit is not Stage12 acceptance.
