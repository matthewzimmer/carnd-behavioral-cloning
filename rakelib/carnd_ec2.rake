CARND_IP = '54.201.139.206'

namespace :carnd do
  task :ssh do
    sh "ssh carnd@#{CARND_IP}"
  end

  task :pull do
    sh "ssh -t carnd@#{CARND_IP} 'cd ~/carnd-behavioral-cloning && git pull'"
  end

  task :train, [:network] => :pull do |t, args|
    args.with_defaults(network: '~/carnd-keras-lab/networks/keras/german_traffic_sign_cnn.py')
    sh "ssh -t carnd@#{CARND_IP} 'python3 #{args[:network]}'"
  end

  task :drive do |t, args|
    sh 'python drive.py model.json'
  end

  task :scp, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{args[:src]} to #{host}:#{args[:dest]}"
    sh "scp -rp #{args[:src]} #{host}:#{args[:dest]}"
  end

  task :sync, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "uploading #{args[:src]} to #{host}:#{args[:dest]}"
    # %w(zimpy networks).each do |file_or_dir|
    #   sh "rsync -avz --exclude '*.zip' --exclude '*.pickle' --exclude '*.p' #{file_or_dir} #{host}:#{args[:dest]}"
    # end

    sh "rsync -ravz --progress driving_log.csv #{host}:~/carnd-behavioral-cloning"
    sh "rsync -ravz --progress --ignore-existing IMG #{host}:~/carnd-behavioral-cloning"

    unless args[:src].nil?
      sh "rsync -avvz --update --existing --ignore-existing #{args[:src]} #{host}:#{args[:dest]}"
    end
  end

  task :get_model, [] do
    host = "carnd@#{CARND_IP}"

    # sh "ssh -t carnd@#{CARND_IP} 'zip -r ~/trained_model.zip carnd-behavioral-cloning/data/trained'"
    # sh "scp -rp carnd@#{CARND_IP}:~/trained_model.zip ."
    # sh 'unzip -u trained_model.zip -d ..'
    sh "rsync -avzh --progress #{host}:~/carnd-behavioral-cloning/data/trained ./data"
  end

  task :down, [:src, :dest] do |t, args|
    args.with_defaults(dest: '~')
    host = "carnd@#{CARND_IP}"
    puts "downloading #{host}:#{args[:src]} to #{args[:dest]}"
    sh "scp -rp #{host}:#{args[:src]} #{args[:dest]}"
  end

  # task :start, [:instance_id] do |t, args|
  #   args.with_defaults(instance_id: INSTANCE_ID)
  #   instance_id = args[:instance_id]
  #   sh "aws ec2 start-instances --instance-ids \"#{instance_id}\""
  # end

  # task :stop, [:instance_id] do |t, args|
  #   args.with_defaults(instance_id: INSTANCE_ID)
  #   instance_id = args[:instance_id]
  #   sh "aws ec2 stop-instances --instance-ids \"#{instance_id}\""
  # end
end
